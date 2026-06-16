#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _ca_reg(void);
extern void _cad_reg(void);
extern void _expsyn2_reg(void);
extern void _expsyn3_reg(void);
extern void _Gfluct_reg(void);
extern void _iahp_reg(void);
extern void _kdr_reg(void);
extern void _ksynProb2_reg(void);
extern void _ksynProb3_reg(void);
extern void _na_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"ca.mod\"");
    fprintf(stderr, " \"cad.mod\"");
    fprintf(stderr, " \"expsyn2.mod\"");
    fprintf(stderr, " \"expsyn3.mod\"");
    fprintf(stderr, " \"Gfluct.mod\"");
    fprintf(stderr, " \"iahp.mod\"");
    fprintf(stderr, " \"kdr.mod\"");
    fprintf(stderr, " \"ksynProb2.mod\"");
    fprintf(stderr, " \"ksynProb3.mod\"");
    fprintf(stderr, " \"na.mod\"");
    fprintf(stderr, "\n");
  }
  _ca_reg();
  _cad_reg();
  _expsyn2_reg();
  _expsyn3_reg();
  _Gfluct_reg();
  _iahp_reg();
  _kdr_reg();
  _ksynProb2_reg();
  _ksynProb3_reg();
  _na_reg();
}

#if defined(__cplusplus)
}
#endif
