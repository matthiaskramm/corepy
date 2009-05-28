#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>
#include <GL/glut.h>
#include <stdio.h>
#include <math.h>

float rtri = 0.0;
float rquad = 0.0;

#define G 6.67428e-11

void c_nb_step(unsigned long ipos, unsigned long ivel, float dt, int n_bodies)
{
  float* pos = (float*)ipos;
  float* vel = (float*)ivel;
  //float* m = (float*)im;
  int i;
  int j;

  for(i = 0; i < n_bodies; i++) {
    double f_x = 0.0;
    double f_y = 0.0;

    for(j = 0; j < n_bodies; j++) {
      double d_x;
      double d_y;
      double dist_tmp;
      double distance;
      double force;

      if(i == j) {
        continue;
      }

      d_x = pos[i * 4] - pos[j * 4];
      d_y = pos[i * 4 + 1] - pos[j * 4 + 1];
      dist_tmp = (d_x * d_x) + (d_y * d_y);
      distance = sqrt(dist_tmp);
      //force = G * ((m[i] * m[j]) / dist_tmp);
      force = G * ((pos[i * 4 + 3] * pos[j * 4 + 3]) / dist_tmp);
      f_x -= force * (d_x / distance);
      f_y -= force * (d_y / distance);
    }

    vel[i * 4] += dt * (f_x / pos[i * 4 + 3]);
    vel[i * 4 + 1] += dt * (f_y / pos[i * 4 + 3]);

    pos[i * 4] += dt * vel[i * 4];
    pos[i * 4 + 1] += dt * vel[i * 4 + 1];
  }
}


void render(unsigned long ix, unsigned long iy, unsigned long im, int n_bodies)
{
  float* x = (float*)ix;
  float* y = (float*)iy;
  float* m = (float*)im;
  int i;

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();                    // Reset The View 
  glTranslatef(0.0, 0.0, -30.0);

  glColor3f(1.0, 1.0, 1.0);

  glPointSize(3);
  for(i = 0; i < n_bodies; i++) {
    //printf("body %f %f\n", x[i], y[i]);
//    glPointSize(m[i] / 500000.0);
    glBegin(GL_POINTS);
    glVertex3f(x[i], y[i], 0.0);
    glEnd();
  }

  glutSwapBuffers();
}


void render2(unsigned long ipos, unsigned long im, int n_bodies)
{
  float* pos = (float*)ipos;
  float* m = (float*)im;
  int i;

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();                    // Reset The View 
  glTranslatef(0.0, 0.0, -30.0);

  glColor3f(1.0, 1.0, 1.0);

  for(i = 0; i < n_bodies; i++) {
    //printf("body %f %f\n", x[i], y[i]);
//    glPointSize(m[i] / 500000.0);
    glBegin(GL_POINTS);
    glVertex3f(pos[i * 4], pos[i * 4 + 1], 0.0);
    glEnd();
  }

  glutSwapBuffers();
}


void render3(unsigned long ipos, int n_bodies)
{
  float* pos = (float*)ipos;
  int i;

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();                    // Reset The View 
  glTranslatef(0.0, 0.0, -30.0);

  glColor3f(1.0, 1.0, 1.0);

  for(i = 0; i < 4; i++) {
    //printf("body %f %f\n", x[i], y[i]);
    glPointSize(9.5);
    glBegin(GL_POINTS);
    glVertex3f(pos[i * 4], pos[i * 4 + 1], 0.0);
    glEnd();
  }


  glPointSize(1.0);
  for(i = 4; i < n_bodies; i++) {
    //printf("body %f %f\n", x[i], y[i]);
    //glPointSize((pos[i * 4 + 3] - 1e7)  * 4.0/ (1e12 - 1e7));
    glBegin(GL_POINTS);
    glVertex3f(pos[i * 4], pos[i * 4 + 1], 0.0);
    glEnd();
  }

  glutSwapBuffers();
}


void render_foo(void)
{
  //printf("got ptr %lu len %d\n", x, len);
  // Clear The Screen And The Depth Buffer
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();                    // Reset The View 

  // Move Left 1.5 units and into the screen 6.0 units.
  glTranslatef(-1.5, 0.0, -6.0);
  glScalef(0.5, 0.5, 0.5);

  // We have smooth color mode on, this will blend across the vertices.
  // Draw a triangle rotated on the Y axis. 
  glRotatef(rtri, 0.0, 1.0, 0.0);      // Rotate
  glBegin(GL_POLYGON);                 // Start drawing a polygon
  glColor3f(1.0, 0.0, 0.0);            // Red
  glVertex3f(0.0, 1.0, 0.0);           // Top
  glColor3f(0.0, 1.0, 0.0);            // Green
  glVertex3f(1.0, -1.0, 0.0);          // Bottom Right
  glColor3f(0.0, 0.0, 1.0);            // Blue
  glVertex3f(-1.0, -1.0, 0.0);         // Bottom Left
  glEnd();                             // We are done with the polygon

  // We are "undoing" the rotation so that we may rotate the quad on its own axis.
  // We also "undo" the prior translate.  This could also have been done using the
  // matrix stack.
  glLoadIdentity();

  // Move Right 1.5 units and into the screen 6.0 units.
  glTranslatef(1.5, 0.0, -6.0);

  // Draw a square (quadrilateral) rotated on the X axis.
  glRotatef(rquad, 1.0, 0.0, 0.0);     // Rotate 
  glColor3f(0.3, 0.5, 1.0);            // Bluish shade
  glBegin(GL_QUADS);                   // Start drawing a 4 sided polygon
  glVertex3f(-1.0, 1.0, 0.0);          // Top Left
  glVertex3f(1.0, 1.0, 0.0);           // Top Right
  glVertex3f(1.0, -1.0, 0.0);          // Bottom Right
  glVertex3f(-1.0, -1.0, 0.0);         // Bottom Left
  glEnd();                             // We are done with the polygon

  // What values to use?  Well, if you have a FAST machine and a FAST 3D Card, then
  // large values make an unpleasant display with flickering and tearing.  I found that
  // smaller values work better, but this was based on my experience.
  rtri  = rtri + 1.0;                  // Increase The Rotation Variable For The Triangle
  rquad = rquad - 1.0;                 // Decrease The Rotation Variable For The Quad

  glutSwapBuffers();
}

int main(int argc, char** argv)
{
  puts("main");
  return 0;
}
