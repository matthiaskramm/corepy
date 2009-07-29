import corepy.lib.extarray as extarray
import corepy.arch.cal.isa as cal
import corepy.arch.cal.types.registers as reg
import corepy.arch.cal.platform as env

import math
import ctypes

G = 6.67428e-11
GPUNUM = 1

glrender = ctypes.CDLL("glrender.so")
glrender.c_nb_step.argtypes = (ctypes.c_ulong, ctypes.c_ulong, ctypes.c_float, ctypes.c_int)
glrender.render.restype = None


def py_nb_step3(x, y, vx, vy, m, dt):
  nbodies = len(x)
  fx = [0.0 for i in xrange(0, nbodies)]
  fy = [0.0 for i in xrange(0, nbodies)]

  for i in xrange(0, nbodies):
    # Forces
    for j in xrange(i + 1, nbodies):
      d_x = x[i] - x[j]
      d_y = y[i] - y[j]
      dist_tmp = d_x**2 + d_y**2
      distance = math.sqrt(dist_tmp)
      force = G * ((m[i] * m[j]) / dist_tmp)
      f_x = force * (d_x / distance)
      f_y = force * (d_y / distance)
      fx[i] -= f_x
      fy[i] -= f_y
      fx[j] += f_x
      fy[j] += f_y

    # Accelerations
    ax = fx[i] / m[i]
    ay = fy[i] / m[i]
 
    # Velocity
    vx[i] += dt * ax
    vy[i] += dt * ay

    # Position
    x[i] += dt * vx[i];
    y[i] += dt * vy[i];

  return

def cal_nb_generate(n_bodies, dt):
  code = env.InstructionStream()
  cal.set_active_code(code)
  fn_bodies = float(n_bodies)

  r_count = code.acquire_register()
  r_lpos = code.acquire_register()
  r_rpos = code.acquire_register()
  r_force = code.acquire_register()
  r_diff = code.acquire_register()
  r_dist_vec = code.acquire_register()
  r_dist = code.acquire_register()
  r_force_tmp = code.acquire_register()
  r_force_vec = code.acquire_register()
  r_vel = code.acquire_register()

  #code.add("dcl_input_position_interp(linear_noperspective) v0.x___")
  cal.dcl_input(reg.v0.x___, USAGE=cal.usage.pos, INTERP=cal.interp.linear_noperspective)
  r_bodies = code.acquire_register((fn_bodies,) * 4)
  r_G = code.acquire_register((G,) * 4)
  r_dt = code.acquire_register((dt,) * 4)
  cal.dcl_output(reg.o0, USAGE=cal.usage.generic)
  cal.dcl_output(reg.o1, USAGE=cal.usage.generic)
  cal.dcl_resource(0, cal.pixtex_type.oned, cal.fmt.float, UNNORM=True) # positions
  cal.dcl_resource(1, cal.pixtex_type.oned, cal.fmt.float, UNNORM=True) # velocities

  # Loop over all other points to calculate the force
  cal.mov(r_count, r_count('0000'))                  # loop counter
  cal.sample(0, 0, r_lpos, reg.v0.x)        # Local position
  cal.mov(r_force, r_force('0000'))                  # total force


  # Compute force using input from every other point
  cal.whileloop()
  # Break if end of points reached
  cal.breakc(cal.relop.ge, r_count, r_bodies)

  cal.sample(0, 0, r_rpos, r_count.x)       # Remote position

  # d_xyz
  cal.sub(r_diff, r_lpos.xyz0, r_rpos.xyz0)   # local pos - remote pos

  # dist_tmp
  cal.mul(r_dist_vec, r_diff.xxxx, r_diff.xxxx)
  cal.mad(r_dist_vec, r_diff.yyyy, r_diff.yyyy, r_dist_vec)
  cal.mad(r_dist_vec, r_diff.zzzz, r_diff.zzzz, r_dist_vec)
  
  # distance
  # TODO - skip rest of force computation if distance is 0
  cal.sqrt_vec(r_dist, r_dist_vec)

  # force G * ((m[i] * m[j]) / dist_tmp)
  cal.mul(r_force_tmp, r_lpos.wwww, r_rpos.wwww)
  cal.div(cal.zeroop.zero, r_force_tmp, r_force_tmp, r_dist_vec)
  cal.mul(r_force_tmp, r_force_tmp, r_G)

  # f_xyz
  cal.div(cal.zeroop.zero, r_force_vec, r_diff.xyz0, r_dist.xyz1)
  cal.mul(r_force_vec, r_force_vec.xyz0, r_force_tmp.xyz0)

  cal.sub(r_force, r_force.xyz0, r_force_vec.xyz0)


  # Increment loop counter, end loop
  cal.add(r_count, r_count, r_count('1111'))
  cal.endloop()

  # Acceleration
  cal.div(cal.zeroop.zero, r_force, r_force.xyz0, r_lpos.wwww)

  # Velocity
  cal.sample(1, 1, r_vel, reg.v0.x)    # Load velocity
  cal.mad(r_vel, r_force, r_dt, r_vel)
  cal.mov(reg.o1, r_vel)

  # Position
  cal.mad(reg.o0, r_vel.xyz0, r_dt.xyz0, r_lpos.xyzw)

  return code


def cal_nb_generate_2d(prgm, n_bodies, dt):
  code = prgm.get_stream()
  cal.set_active_code(code)
  fn_bodies = float(n_bodies)

  #r_cx = prgm.acquire_register()
  #r_cy = prgm.acquire_register()
  r_count = prgm.acquire_register()
  r_lpos = prgm.acquire_register()
  r_rpos = prgm.acquire_register()
  r_force = prgm.acquire_register()
  r_diff = prgm.acquire_register()
  r_dist_vec = prgm.acquire_register()
  r_dist = prgm.acquire_register()
  r_force_tmp = prgm.acquire_register()
  r_force_vec = prgm.acquire_register()
  r_vel = prgm.acquire_register()

  #code.add("dcl_input_position_interp(linear_noperspective) v0.xy__")
  cal.dcl_input(reg.v0.x___, USAGE=cal.usage.pos, INTERP=cal.interp.linear_noperspective)
  r_bodies = prgm.acquire_register((fn_bodies,) * 4)
  r_G = prgm.acquire_register((G,) * 4)
  r_dt = prgm.acquire_register((dt,) * 4)
  cal.dcl_output(reg.o0, USAGE=cal.usage.generic)
  cal.dcl_output(reg.o1, USAGE=cal.usage.generic)
  cal.dcl_resource(0, cal.pixtex_type.twod, cal.fmt.float, UNNORM=True) # positions
  cal.dcl_resource(1, cal.pixtex_type.twod, cal.fmt.float, UNNORM=True) # velocities

  # Loop over all other points to calculate the force
  cal.mov(r_count, r_count('0000'))                  # loop counter
  #cal.mov(r_cx, r_cx('0000'))                  # loop counter
  #cal.mov(r_cy, r_cy('0000'))                  # loop counter
  cal.sample(0, 0, r_lpos, reg.v0.xy)                # Local position
  cal.mov(r_force, r_force('0000'))                  # total force


  # Compute force using input from every other point
  cal.whileloop()
  cal.breakc(cal.relop.ge, r_count.x, r_bodies)

  cal.mov(r_count, r_count.x0zw)

  cal.whileloop()
  cal.breakc(cal.relop.ge, r_count.y, r_bodies)

  #for i in xrange(0, 4):
  #cal.add(r_count, r_cx('x000'), r_cy('0x00'))
  cal.sample(0, 0, r_rpos, r_count.xy)       # Remote position

  # d_xyz
  cal.sub(r_diff, r_lpos.xyz0, r_rpos.xyz0)   # local pos - remote pos

  # dist_tmp
  #cal.mul(r_dist_vec, r_diff.xxxx, r_diff.xxxx)
  #cal.mad(r_dist_vec, r_diff.yyyy, r_diff.yyyy, r_dist_vec)
  #cal.mad(r_dist_vec, r_diff.zzzz, r_diff.zzzz, r_dist_vec)
  cal.dp3(r_dist_vec, r_diff, r_diff, IEEE = False)
  
  # distance
  # TODO - skip rest of force computation if distance is 0
  cal.sqrt_vec(r_dist, r_dist_vec)

  # force G * ((m[i] * m[j]) / dist_tmp)
  cal.mul(r_force_tmp, r_lpos.wwww, r_rpos.wwww, IEEE = False)
  cal.div(r_force_tmp, r_force_tmp, r_dist_vec, ZEROOP = cal.zeroop.zero)
  cal.mul(r_force_tmp, r_force_tmp, r_G, IEEE = False)

  # f_xyz
  # TODO - whats going on, is this right?
  cal.div(r_force_vec, r_diff.xyz0, r_dist.xyz1, ZEROOP = cal.zeroop.zero)
  cal.mul(r_force_vec, r_force_vec.xyz0, r_force_tmp.xyz0, IEEE = False)

  cal.sub(r_force, r_force.xyz0, r_force_vec.xyz0)

  #cal.add(r_cy, r_cy, r_count('1111'))

  #cal.add(r_count, r_count, r_count('0100'))
  #cal.ifc(cal.relop.ge, r_count.y, r_bodies.y)
  ## TODO - can I merge these two?
  #cal.mov(r_count('_y__'), r_count('x0zw'))
  #cal.add(r_count, r_count, r_count('1000'))
  #cal.endif()

  # Increment loop counter, end loop
  cal.add(r_count, r_count, r_count('0100'))
  cal.endloop()

  cal.add(r_count, r_count, r_count('1000'))
  #cal.add(r_cx, r_cx, r_cx('1111'))
  cal.endloop()

  # Acceleration
  cal.div(r_force, r_force.xyz0, r_lpos.wwww, ZEROOP = cal.zeroop.zero)

  # Velocity
  cal.sample(1, 1, r_vel, reg.v0.xy)    # Load velocity
  cal.mad(r_vel, r_force, r_dt, r_vel, IEEE = False)
  cal.mov(reg.o1, r_vel)

  # Position
  cal.mad(reg.o0, r_vel.xyz0, r_dt.xyz0, r_lpos.xyzw, IEEE = False)

  #cal.mov(reg.g[0], r_vel)

  return code


def cal_nb_generate_local(n_bodies, dt, steps):
  code = env.InstructionStream()
  cal.set_active_code(code)
  fn_bodies = float(n_bodies)
  steps = float(steps)

  r_count = code.acquire_register()
  r_step = code.acquire_register()
  r_lpos = code.acquire_register()
  r_rpos = code.acquire_register()
  r_force = code.acquire_register()
  r_diff = code.acquire_register()
  r_dist_vec = code.acquire_register()
  r_dist = code.acquire_register()
  r_force_tmp = code.acquire_register()
  r_force_vec = code.acquire_register()
  r_vel = code.acquire_register()

  print "fn_bodies", fn_bodies

  code.add("dcl_input_position_interp(linear_noperspective) v0.xy__")
  #cal.dcl_input(reg.v0.x___, USAGE=cal.usage.pos, INTERP=cal.interp.linear_noperspective)
  r_numsteps = code.acquire_register((steps,) * 4)
  r_bodies = code.acquire_register((fn_bodies,) * 4)
  #r_bodiesquare = code.acquire_register((float(fn_bodies**2),) * 4)
  r_G = code.acquire_register((G,) * 4)
  r_dt = code.acquire_register((dt,) * 4)
  cal.dcl_output(reg.o0, USAGE=cal.usage.generic)
  cal.dcl_output(reg.o1, USAGE=cal.usage.generic)
  cal.dcl_output(reg.o2, USAGE=cal.usage.generic)
  cal.dcl_resource(0, cal.pixtex_type.twod, cal.fmt.float, UNNORM=True) # positions
  cal.dcl_resource(1, cal.pixtex_type.twod, cal.fmt.float, UNNORM=True) # velocities

  r_foo = code.acquire_register()
  cal.mov(r_foo, r_foo('0000'))

  r_gpos = code.acquire_register()
  cal.mad(r_gpos, reg.v0.y, r_bodies.x, reg.v0.x)

  r_gvel = code.acquire_register()
  cal.mad(r_gvel, r_bodies.x, r_bodies.x, r_gpos)

  cal.ftoi(r_gpos, r_gpos)
  cal.ftoi(r_gvel, r_gvel)

  cal.sample(0, 0, r_lpos, reg.v0.xy)                # Local position
  cal.sample(1, 1, r_vel, reg.v0.xy)    # Load velocity

  cal.mov(reg.g[r_gpos.x], r_lpos)
  cal.mov(reg.g[r_gvel.x], r_vel)

  cal.mov(r_step, r_step('0000'))

  cal.whileloop()
  cal.breakc(cal.relop.ge, r_step.x, r_numsteps)

  cal.mov(r_count, r_count('0000'))                  # loop counter

  cal.whileloop()
  cal.breakc(cal.relop.ge, r_count.x, r_bodies)

  cal.add(r_foo, r_foo, r_foo('1111'))

  # calculate force
  r_tmp = code.acquire_register()
  cal.ftoi(r_tmp, r_count)

  cal.mov(r_rpos, reg.g[r_tmp.x])

  # d_xyz
  cal.sub(r_diff, r_lpos.xyz0, r_rpos.xyz0)   # local pos - remote pos

  # dist_tmp
  cal.mul(r_dist_vec, r_diff.xxxx, r_diff.xxxx)
  cal.mad(r_dist_vec, r_diff.yyyy, r_diff.yyyy, r_dist_vec)
  cal.mad(r_dist_vec, r_diff.zzzz, r_diff.zzzz, r_dist_vec)
  
  # distance
  # TODO - skip rest of force computation if distance is 0
  cal.sqrt_vec(r_dist, r_dist_vec)

  # force G * ((m[i] * m[j]) / dist_tmp)
  cal.mul(r_force_tmp, r_lpos.wwww, r_rpos.wwww)
  cal.div(r_force_tmp, r_force_tmp, r_dist_vec, ZEROOP = cal.zeroop.zero)
  cal.mul(r_force_tmp, r_force_tmp, r_G)

  # f_xyz
  # TODO - whats going on, is this right?
  cal.div(r_force_vec, r_diff.xyz0, r_dist.xyz1, ZEROOP = cal.zeroop.zero)
  cal.mul(r_force_vec, r_force_vec.xyz0, r_force_tmp.xyz0)

  cal.sub(r_force, r_force.xyz0, r_force_vec.xyz0)

  cal.add(r_count, r_count, r_count('1111'))
  cal.endloop()

  # Acceleration
  cal.div(r_force, r_force.xyz0, r_lpos.wwww, ZEROOP = cal.zeroop.zero)

  # Velocity
  cal.mad(r_vel, r_force, r_dt, r_vel)

  # Position
  cal.mad(reg.o0, r_vel.xyz0, r_dt.xyz0, r_lpos.xyzw)

  # store updated pos and vel
  cal.mov(reg.g[r_gpos.x], r_lpos)
  cal.mov(reg.g[r_gvel.x], r_vel)

  cal.add(r_step, r_step, r_step('1111'))
  cal.endloop()

  cal.mov(reg.o0, r_lpos)
  cal.mov(reg.o1, r_vel)
  cal.mov(reg.o2, r_foo)
  return code






def cal_nb_exec(proc, code, ipos, ivel, opos, ovel):
  code.set_remote_binding(reg.i0, ipos)
  code.set_remote_binding(reg.i1, ivel)
  code.set_remote_binding(reg.o0, opos)
  code.set_remote_binding(reg.o1, ovel)

  domain = (0, 0, len(ipos) / 4, 1)

  proc.execute(code, domain)
  return


if __name__ == '__main__':
  import time
  import random
  proc = env.Processor(GPUNUM)
  #code = env.InstructionStream()
  #proc = env.Processor()
  SQRT_NBODIES = 64
  N_BODIES = SQRT_NBODIES * SQRT_NBODIES
  DT = 0.25
  STEPS = 100
  random.seed(0)

  #init_x = (3.0e11, 5.79e10, 1.082e11, 1.496e11, 2.279e11)
  #init_yv = (0.0, 2.395e4, 1.75e4, 1.49e4, 1.205e4)
  #init_m = (1.989e30, 3.302e23, 4.869e24, 5.974e24, 6.419e23)

  x = [random.uniform(-1.0, 1.0) for i in xrange(0, N_BODIES)]
  y = [random.uniform(-1.0, 1.0) for i in xrange(0, N_BODIES)]
  vx = [random.uniform(-3.0, 3.0) for i in xrange(0, N_BODIES)]
  vy = [random.uniform(-3.0, 3.0) for i in xrange(0, N_BODIES)]
  m = [random.uniform(1e3, 1e4) for i in xrange(0, N_BODIES)]

  bx = x[:]
  by = y[:]
  bvx = vx[:]
  bvy = vy[:]
  bm = m[:]

  #t1 = time.time()
  #for i in xrange(0, STEPS, 1):
  #  py_nb_step3(bx, by, bvx, bvy, bm, DT)
  #t2 = time.time()

  #print "py_nb_step3 time", t2 - t1

  #debug = proc.alloc_remote('f', 4, SQRT_NBODIES, SQRT_NBODIES)
  pos = [proc.alloc_remote('f', 4, SQRT_NBODIES, SQRT_NBODIES) for i in xrange(0, 2)]
  vel = [proc.alloc_remote('f', 4, SQRT_NBODIES, SQRT_NBODIES) for i in xrange(0, 2)]
  localpos = [proc.alloc_local('f', 4, SQRT_NBODIES, SQRT_NBODIES) for i in xrange(0, 2)]
  localvel = [proc.alloc_local('f', 4, SQRT_NBODIES, SQRT_NBODIES) for i in xrange(0, 2)]

  if pos[0].gpu_pitch != SQRT_NBODIES:
    print "WARNING pitch is not the same as the width!", pos[0].gpu_pitch, SQRT_NBODIES

  pos[0].clear()
  vel[0].clear()
  pos[1].clear()
  vel[1].clear()
  for i in xrange(0, N_BODIES):
    #pos[0][i * 4] = x[i]
    #pos[0][i * 4 + 1] = y[i]
    pos[0][i * 4] = float(i - (N_BODIES / 2))
    pos[0][i * 4 + 1] = 0.0
    pos[0][i * 4 + 2] = 0.0
    pos[0][i * 4 + 3] = 100000.0
    #pos[0][i * 4 + 3] = m[i] # mass is here
    #vel[0][i * 4] = vx[i]
    #vel[0][i * 4 + 1] = vy[i]
    vel[0][i * 4 + 0] = 0.0
    vel[0][i * 4 + 1] = 0.0
    vel[0][i * 4 + 2] = 0.0
    vel[0][i * 4 + 3] = 0.0

  prgm = env.Program()
  code = cal_nb_generate_2d(prgm, SQRT_NBODIES, DT)
  for i in xrange(0, 25):
    prgm.add(code)
  prgm.cache_code()
  #print code.render_string

  #code.set_local_binding('g[]', (SQRT_NBODIES, SQRT_NBODIES * 2, env.cal_exec.FMT_FLOAT32_4))

  t1 = time.time()
  proc.copy(localpos[0], pos[0])
  proc.copy(localvel[0], vel[0])

  for i in xrange(0, STEPS, 1):
    #print "STEPS", i
    inp = i % 2
    out = (i + 1) % 2
    #cal_nb_exec(proc, code, pos[inp], vel[inp], pos[out], vel[out])
    #glrender.c_nb_step(pos[0].buffer_info()[0], vel[0].buffer_info()[0], DT, N_BODIES)
    prgm.set_binding(reg.i0, localpos[inp])
    prgm.set_binding(reg.i1, localvel[inp])
    prgm.set_binding(reg.o0, localpos[out])
    prgm.set_binding(reg.o1, localvel[out])

    domain = (0, 0, SQRT_NBODIES, SQRT_NBODIES)

    #t3 = time.time()
    proc.execute(prgm, domain)
    #t4 = time.time()
    #print "step %d time %f" % (i, t4 - t3)

  cpy1 = proc.copy(pos[out], localpos[out], async = True)
  cpy2 = proc.copy(vel[out], localvel[out], async = True)

  proc.join(cpy1)
  proc.join(cpy2)
  t2 = time.time()

  print "cal_nb_exec time", t2 - t1

  total = 0.0
  for i in xrange(0, N_BODIES):
    diff = abs(pos[out][i * 4] - bx[i])
    diff += abs(pos[out][i * 4 + 1] - by[i])
    #print "y py %f cal %f diff %f" % (by[i], pos[0][i * 4 + 1], diff)
    total += diff
  print "total diff", total

  for i in xrange(0, 16):
    print "%d %10.8f %10.8f" % (i, pos[out][i * 4], vel[out][i * 4])
  print
  for i in xrange(N_BODIES - 16, N_BODIES):
    print "%d %10.8f %10.8f" % (i, pos[out][i * 4], vel[out][i * 4])

  for i in xrange(0, 2):
    proc.free(pos[i])
    proc.free(vel[i])

