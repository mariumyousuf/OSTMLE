#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

extern void _Gfluct_reg();
extern void _RCHAN_reg();
extern void _ca_reg();
extern void _cad_reg();
extern void _expsyn2_reg();
extern void _expsyn3_reg();
extern void _iahp_reg();
extern void _kdr_reg();
extern void _ksynProb2_reg();
extern void _ksynProb3_reg();
extern void _na_reg();

void modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," Gfluct.mod");
fprintf(stderr," RCHAN.MOD");
fprintf(stderr," ca.mod");
fprintf(stderr," cad.mod");
fprintf(stderr," expsyn2.mod");
fprintf(stderr," expsyn3.mod");
fprintf(stderr," iahp.mod");
fprintf(stderr," kdr.mod");
fprintf(stderr," ksynProb2.mod");
fprintf(stderr," ksynProb3.mod");
fprintf(stderr," na.mod");
fprintf(stderr, "\n");
    }
_Gfluct_reg();
_RCHAN_reg();
_ca_reg();
_cad_reg();
_expsyn2_reg();
_expsyn3_reg();
_iahp_reg();
_kdr_reg();
_ksynProb2_reg();
_ksynProb3_reg();
_na_reg();
}
