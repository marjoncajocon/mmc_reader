/*
** mrnet.h
** Neural network: a small multi-layer perceptron, its training and its
** model file
** See Copyright Notice in mr.h
*/

#ifndef mrnet_h
#define mrnet_h

#include <stdint.h>

#include "mrlimits.h"
#include "mrrand.h"


#define MR_MAXLAYERS  4

/* extension of model files: <lang>.mrm */
#define MR_MODELEXT  ".mrm"


/* code point of the "not one letter" class (two glued letters, junk) */
#define MR_REJECT  0

/*
** Layer 'l' maps sizes[l] inputs to sizes[l + 1] outputs. Hidden layers
** use ReLU, the last one softmax. Output 'k' is the letter classes[k].
*/
typedef struct mr_Net {
  int nlayers;
  int sizes[MR_MAXLAYERS + 1];
  float *w[MR_MAXLAYERS];  /* sizes[l + 1] rows of sizes[l] weights */
  float *b[MR_MAXLAYERS];  /* sizes[l + 1] biases */
  uint32_t *classes;  /* Unicode code point of each output */
} mr_Net;


/* training state: gradients, Adam moments, scratch space */
typedef struct mr_Trainer mr_Trainer;


MRI_FUNC int mrN_new (mr_State *R, int nlayers, const int *sizes,
                      mr_Net **out);
MRI_FUNC void mrN_free (mr_State *R, mr_Net *net);
MRI_FUNC void mrN_randomize (mr_Net *net, mr_Rand *rng);
MRI_FUNC int mrN_nclasses (const mr_Net *net);

/* number of floats of scratch space 'mrN_classify' needs */
MRI_FUNC size_t mrN_worksize (const mr_Net *net);

/*
** Run the network on 'in'. Returns the best class index and puts its
** probability in '*conf' (if not NULL). 'work' has mrN_worksize floats.
*/
MRI_FUNC int mrN_classify (const mr_Net *net, const float *in, float *work,
                           float *conf);

/* the class probabilities of the last 'mrN_classify' with 'work' */
MRI_FUNC const float *mrN_probs (const mr_Net *net, const float *work);

MRI_FUNC int mrN_load (mr_State *R, const char *path, mr_Net **out);
MRI_FUNC int mrN_save (mr_State *R, const mr_Net *net, const char *path);

MRI_FUNC int mrN_newtrainer (mr_State *R, const mr_Net *net,
                             mr_Trainer **out);
MRI_FUNC void mrN_freetrainer (mr_State *R, mr_Trainer *t);

/*
** One training step on 'n' samples ('in' holds n * sizes[0] floats).
** Returns the mean loss; '*correct' gets how many were already right.
*/
MRI_FUNC float mrN_train (mr_Trainer *t, mr_Net *net, const float *in,
                          const int *labels, int n, float rate,
                          int *correct);

#endif
