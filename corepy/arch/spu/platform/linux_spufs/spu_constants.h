
#ifndef SPU_CONSTANTS_H
#define SPU_CONSTANTS_H

// Constants for managing the SPU

// SPU_READY is the same as the context switching code's SPU_SAVE_COMPLETE tag
// Scratch that... let's use something that starts with 0
#define SPU_READY 0x000C

// From BE Handbook p. 422
#define SPU_EXIT_FAILURE 0x2001


// From spu.h

typedef unsigned short u16;
typedef unsigned char  u8;
typedef unsigned int   u32;
typedef unsigned long  u64;

#define MFC_PUT_CMD             0x20
#define MFC_PUTS_CMD            0x28
#define MFC_PUTR_CMD            0x30
#define MFC_PUTF_CMD            0x22
#define MFC_PUTB_CMD            0x21
#define MFC_PUTFS_CMD           0x2A
#define MFC_PUTBS_CMD           0x29
#define MFC_PUTRF_CMD           0x32
#define MFC_PUTRB_CMD           0x31
#define MFC_PUTL_CMD            0x24
#define MFC_PUTRL_CMD           0x34
#define MFC_PUTLF_CMD           0x26
#define MFC_PUTLB_CMD           0x25
#define MFC_PUTRLF_CMD          0x36
#define MFC_PUTRLB_CMD          0x35

#define MFC_GET_CMD             0x40
#define MFC_GETS_CMD            0x48
#define MFC_GETF_CMD            0x42
#define MFC_GETB_CMD            0x41
#define MFC_GETFS_CMD           0x4A
#define MFC_GETBS_CMD           0x49
#define MFC_GETL_CMD            0x44
#define MFC_GETLF_CMD           0x46
#define MFC_GETLB_CMD           0x45

#define MFC_SDCRT_CMD           0x80
#define MFC_SDCRTST_CMD         0x81
#define MFC_SDCRZ_CMD           0x89
#define MFC_SDCRS_CMD           0x8D
#define MFC_SDCRF_CMD           0x8F

#define MFC_GETLLAR_CMD         0xD0
#define MFC_PUTLLC_CMD          0xB4
#define MFC_PUTLLUC_CMD         0xB0
#define MFC_PUTQLLUC_CMD        0xB8
#define MFC_SNDSIG_CMD          0xA0
#define MFC_SNDSIGB_CMD         0xA1
#define MFC_SNDSIGF_CMD         0xA2
#define MFC_BARRIER_CMD         0xC0
#define MFC_EIEIO_CMD           0xC8
#define MFC_SYNC_CMD            0xCC

#define MFC_MIN_DMA_SIZE_SHIFT  4       /* 16 bytes */
#define MFC_MAX_DMA_SIZE_SHIFT  14      /* 16384 bytes */
#define MFC_MIN_DMA_SIZE        (1 << MFC_MIN_DMA_SIZE_SHIFT)
#define MFC_MAX_DMA_SIZE        (1 << MFC_MAX_DMA_SIZE_SHIFT)
#define MFC_MIN_DMA_SIZE_MASK   (MFC_MIN_DMA_SIZE - 1)
#define MFC_MAX_DMA_SIZE_MASK   (MFC_MAX_DMA_SIZE - 1)
#define MFC_MIN_DMA_LIST_SIZE   0x0008  /*   8 bytes */
#define MFC_MAX_DMA_LIST_SIZE   0x4000  /* 16K bytes */

#define MFC_TAGID_TO_TAGMASK(tag_id)  (1 << (tag_id & 0x1F))

#endif // SPU_CONSTANTS_H
