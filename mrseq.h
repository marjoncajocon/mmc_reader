/*
** mrseq.h
** Line recognizer for handwriting: CNN + bidirectional LSTM + CTC.
** Reads a whole text line at once, so joined (cursive) letters need no
** cutting.
** See Copyright Notice in mr.h
*/

#ifndef mrseq_h
#define mrseq_h

#include <stdint.h>

#include "mrlimits.h"
#include "mrbuf.h"
#include "mrrand.h"


#define MR_SEQH  32  /* line height fed to the network */
#define MR_SEQMAXW  2048  /* widest line, in pixels after scaling */
#define MR_SEQSTEP  4  /* pixels per output step (two 2x poolings) */
#define MR_SEQBLANK  0  /* class 0 is the CTC blank */


/*
** The network. Every weight lives in one array 'p' (so training, saving
** and loading are simple); 'off' says where each part starts.
*/
typedef struct mr_Seq {
  int nclasses;  /* output classes, blank included */
  uint32_t *classes;  /* code point of each class; classes[0] = blank */
  int hidden;  /* LSTM cells per direction */
  float *p;  /* all weights */
  size_t np;  /* number of weights */
} mr_Seq;


/* scratch space and gradients for one line; one per thread */
typedef struct mr_SeqWork mr_SeqWork;


MRI_FUNC int mrQ_new (mr_State *R, int nclasses, int hidden, mr_Seq **out);
MRI_FUNC void mrQ_free (mr_State *R, mr_Seq *q);
MRI_FUNC void mrQ_randomize (mr_Seq *q, mr_Rand *rng);
MRI_FUNC int mrQ_load (mr_State *R, const char *path, mr_Seq **out);
MRI_FUNC int mrQ_save (mr_State *R, const mr_Seq *q, const char *path);

/* 1 if file 'path' holds a line model (not a letter model) */
MRI_FUNC int mrQ_isseq (const char *path);

MRI_FUNC int mrQ_newwork (mr_State *R, const mr_Seq *q, mr_SeqWork **out);
MRI_FUNC void mrQ_freework (mr_State *R, mr_SeqWork *w);

/*
** Scale ink 'bits' (w x h bytes, 1 = ink) to MR_SEQH rows, cropped to
** the ink. 'line' gets MR_SEQH x MR_SEQMAXW floats; returns the width
** used (0 if there is no ink).
*/
MRI_FUNC int mrQ_normalize (const mr_byte *bits, int w, int h,
                            float stretch, float *line);

/* read a normalized line; adds UTF-8 text to 'out' */
MRI_FUNC int mrQ_read (mr_State *R, const mr_Seq *q, mr_SeqWork *wk,
                       const float *line, int width, mr_Buffer *out);

/*
** Training: forward + CTC + backward for one line with target classes
** 'target[0..n-1]'. Adds to the gradients in 'wk' and returns the loss
** (negative log likelihood), or a negative number if the line is too
** short for its text.
*/
MRI_FUNC float mrQ_learn (const mr_Seq *q, mr_SeqWork *wk, const float *line,
                          int width, const int *target, int n);

/* gradients of 'wk' (cleared after use) */
MRI_FUNC float *mrQ_grad (mr_SeqWork *wk);
MRI_FUNC void mrQ_cleargrad (const mr_Seq *q, mr_SeqWork *wk);

#endif
