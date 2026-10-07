/*
** mrglyph.h
** Letter segment -> neural network input
** See Copyright Notice in mr.h
*/

#ifndef mrglyph_h
#define mrglyph_h

#include "mrlimits.h"
#include "mrlayout.h"


/* the letter shape, scaled to fit a square, keeping its aspect */
#define MR_GLYPHSIZE  32

/*
** Size and place of the letter in its line, in x-heights. The shape
** alone cannot tell 'o' from 'O', ',' from '\'' or '-' from '_'.
*/
#define MR_GLYPHFEAT  5

#define MR_GLYPHINPUT  (MR_GLYPHSIZE * MR_GLYPHSIZE + MR_GLYPHFEAT)


/*
** Fill 'in' (MR_GLYPHINPUT floats) for the ink of segments 'sega' to
** 'segb' (both included; -1 for 'segb' means only 'sega') inside 'box',
** on line 'ln'.
*/
MRI_FUNC void mrG_extract (const mr_Layout *lo, const mr_Line *ln,
                           const mr_Box *box, long sega, long segb,
                           float *in);

#endif
