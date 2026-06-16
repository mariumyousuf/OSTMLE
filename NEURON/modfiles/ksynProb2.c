/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scoplib.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__KSynProb
#define _nrn_initial _nrn_initial__KSynProb
#define nrn_cur _nrn_cur__KSynProb
#define _nrn_current _nrn_current__KSynProb
#define nrn_jacob _nrn_jacob__KSynProb
#define nrn_state _nrn_state__KSynProb
#define _net_receive _net_receive__KSynProb 
#define conductance conductance__KSynProb 
#define initvars initvars__KSynProb 
#define state state__KSynProb 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define tau _p[0]
#define stim _p[1]
#define e _p[2]
#define gmax _p[3]
#define ntimes _p[4]
#define F0 _p[5]
#define Fmag _p[6]
#define D0 _p[7]
#define Dmag _p[8]
#define from _p[9]
#define compart _p[10]
#define distance _p[11]
#define synlocation _p[12]
#define i _p[13]
#define onset (_p + 14)
#define Ridx _p[2014]
#define treleased (_p + 2015)
#define fromlist (_p + 4015)
#define NReleased _p[6015]
#define p _p[6016]
#define p_release _p[6017]
#define A _p[6018]
#define G _p[6019]
#define index _p[6020]
#define bath _p[6021]
#define k _p[6022]
#define F _p[6023]
#define D _p[6024]
#define DA _p[6025]
#define DG _p[6026]
#define _g _p[6027]
#define _nd_area  *_ppvar[0]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_conductance(void*);
 static double _hoc_initvars(void*);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(Object* _ho) { void* create_point_process(int, Object*);
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt(void*);
 static double _hoc_loc_pnt(void* _vptr) {double loc_point_process(int, void*);
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(void* _vptr) {double has_loc_point(void*);
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(void* _vptr) {
 double get_loc_point_process(void*); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 "conductance", _hoc_conductance,
 "initvars", _hoc_initvars,
 0, 0
};
 /* declare global and static user variables */
#define tauD2 tauD2_KSynProb
 double tauD2 = 9200;
#define tauD1 tauD1_KSynProb
 double tauD1 = 280;
#define tauF tauF_KSynProb
 double tauF = 94;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "tau", "ms",
 "stim", "umho",
 "e", "mV",
 "A", "umho",
 "G", "umho",
 "i", "nA",
 "onset", "ms",
 0,0
};
 static double A0 = 0;
 static double G0 = 0;
 static double delta_t = 1;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "tauF_KSynProb", &tauF_KSynProb,
 "tauD1_KSynProb", &tauD1_KSynProb,
 "tauD2_KSynProb", &tauD2_KSynProb,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 
static int _ode_count(int);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"KSynProb",
 "tau",
 "stim",
 "e",
 "gmax",
 "ntimes",
 "F0",
 "Fmag",
 "D0",
 "Dmag",
 "from",
 "compart",
 "distance",
 "synlocation",
 0,
 "i",
 "onset[2000]",
 "Ridx",
 "treleased[2000]",
 "fromlist[2000]",
 "NReleased",
 "p",
 "p_release",
 0,
 "A",
 "G",
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 6028, _prop);
 	/*initialize range parameters*/
 	tau = 0.25;
 	stim = 50;
 	e = 0;
 	gmax = 0.1;
 	ntimes = 0;
 	F0 = 1;
 	Fmag = 0.1;
 	D0 = 1;
 	Dmag = 0.1;
 	from = -1;
 	compart = 0;
 	distance = 0;
 	synlocation = 0.5;
  }
 	_prop->param = _p;
 	_prop->param_size = 6028;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 2, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _ksynProb2_reg() {
	int _vectorized = 0;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 0,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 6028, 2);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
 	hoc_register_cvode(_mechtype, _ode_count, 0, 0, 0);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 KSynProb ksynProb2.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int conductance();
static int initvars();
 extern double *_getelm();
 
#define _MATELM1(_row,_col)	*(_getelm(_row + 1, _col + 1))
 
#define _RHS1(_arg) _coef1[_arg + 1]
 static double *_coef1;
 
#define _linmat1  1
 static void* _sparseobj1;
 static int _slist1[2], _dlist1[2]; static double *_temp1;
 static int state();
 
static int  initvars (  ) {
   double _li ;
 _li = 0.0 ;
   while ( _li < 2000.0 ) {
     onset [ ((int) _li ) ] = 1e20 ;
     treleased [ ((int) _li ) ] = 0.0 ;
     fromlist [ ((int) _li ) ] = - 1.0 ;
     _li = _li + 1.0 ;
     }
    return 0; }
 
static double _hoc_initvars(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r = 1.;
 initvars (  );
 return(_r);
}
 
static int  conductance (  ) {
   double _lii , _lj ;
 while ( index < 2000.0  && t > onset [ ((int) index ) ] ) {
     F = F0 ;
     _lii = 0.0 ;
     while ( _lii < index ) {
       F = F + Fmag * exp ( - ( t - onset [ ((int) _lii ) ] ) / tauF ) ;
       _lii = _lii + 1.0 ;
       }
     D = D0 ;
     _lii = 0.0 ;
     while ( _lii < Ridx ) {
       D = D - Dmag * exp ( - ( t - treleased [ ((int) _lii ) ] ) / tauD1 ) ;
       _lii = _lii + 1.0 ;
       }
     if ( D < 0.0 ) {
       D = 0.0 ;
       }
     p_release = 1.0 - exp ( - F * D ) ;
     p = scop_random ( 0.0 ) ;
     if ( p < p_release ) {
       from = fromlist [ ((int) index ) ] ;
       
/*VERBATIM*/
	  	fprintf(stderr,"\n Realeased from %.0f onto %.0f at t=%.2f with weight %f",from, compart, t, gmax);
 treleased [ ((int) Ridx ) ] = onset [ ((int) index ) ] ;
       Ridx = Ridx + 1.0 ;
       NReleased = NReleased + 1.0 ;
       A = stim ;
       }
     else {
       A = 0.0 ;
       }
     index = index + 1.0 ;
     }
   error = sparse(&_sparseobj1, 2, _slist1, _dlist1, _p, &t, dt, state,&_coef1, _linmat1);
 if(error){fprintf(stderr,"at line 168 in file ksynProb2.mod:\n	\n"); nrn_complain(_p); abort_run(error);}
    if (secondorder) {
    int _i;
    for (_i = 0; _i < 2; ++_i) {
      _p[_slist1[_i]] += dt*_p[_dlist1[_i]];
    }}
 
/*VERBATIM*/
	return 0;
  return 0; }
 
static double _hoc_conductance(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r = 1.;
 conductance (  );
 return(_r);
}
 
static int state ()
 {_reset=0;
 {
   double b_flux, f_flux, _term; int _i;
 {int _i; double _dt1 = 1.0/dt;
for(_i=0;_i<2;_i++){
  	_RHS1(_i) = -_dt1*(_p[_slist1[_i]] - _p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
} }
 /* ~ A <-> G ( k , 0.0 )*/
 f_flux =  k * A ;
 b_flux =  0.0 * G ;
 _RHS1( 0) -= (f_flux - b_flux);
 _RHS1( 1) += (f_flux - b_flux);
 
 _term =  k ;
 _MATELM1( 0 ,0)  += _term;
 _MATELM1( 1 ,0)  -= _term;
 _term =  0.0 ;
 _MATELM1( 0 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ G <-> bath ( k , 0.0 )*/
 f_flux =  k * G ;
 b_flux =  0.0 * bath ;
 _RHS1( 1) -= (f_flux - b_flux);
 
 _term =  k ;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
    } return _reset;
 }
 
static int _ode_count(int _type){ hoc_execerror("KSynProb", "cannot be used with CVODE"); return 0;}

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  A = A0;
  G = G0;
 {
   k = 1.0 / tau ;
   A = 0.0 ;
   G = 0.0 ;
   index = 0.0 ;
   Ridx = 0.0 ;
   NReleased = 0.0 ;
   p = 0.0 ;
   F = F0 ;
   D = D0 ;
   p_release = 0.0 ;
   initvars ( _threadargs_ ) ;
   index = 0.0 ;
   ntimes = 0.0 ;
   }
  _sav_indep = t; t = _save;

}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel();
}}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   i = gmax * G * ( v - e ) ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _g = _nrn_current(_v + .001);
 	{ _rhs = _nrn_current(_v);
 	}
 _g = (_g - _rhs)/.001;
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
double _dtsav = dt;
if (secondorder) { dt *= 0.5; }
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
 { error =  conductance();
 if(error){fprintf(stderr,"at line 118 in file ksynProb2.mod:\n	SOLVE conductance\n"); nrn_complain(_p); abort_run(error);}
 }}}
 dt = _dtsav;
}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(A) - _p;  _dlist1[0] = &(DA) - _p;
 _slist1[1] = &(G) - _p;  _dlist1[1] = &(DG) - _p;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "ksynProb2.mod";
static const char* nmodl_file_text = 
  "COMMENT\n"
  "alpha function synapse implemented as continuously integrated\n"
  "kinetic scheme a la Srinivasan and Chiel (Neural Computation) so that\n"
  "one can give many stimuli which summate.\n"
  "\n"
  "Onset times are placed in the vector onset[SIZE]\n"
  "Conductance located in state variable G\n"
  "The amplitude of each individual alpha function is given by stim,\n"
  "stim * t * exp(-t/tau).\n"
  "The last onset time should be a very large number so stim stops getting\n"
  "added to state A\n"
  "\n"
  "Jean-Marc: \n"
  "-Included probabilistic release, facilitation and short-term depression, a la Maas&Zador.\n"
  "-Included presynaptic NetCon trigger\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  "DEFINE SIZE 2000\n"
  "\n"
  "INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}\n"
  "\n"
  "\n"
  "NEURON {\n"
  "	POINT_PROCESS KSynProb\n"
  "	RANGE tau, stim, e, i,onset, treleased, fromlist, from, gmax, p,p_release,compart,ntimes,distance,synlocation,D0,F0,Dmag,Fmag, Ridx, NReleased\n"
  "	NONSPECIFIC_CURRENT i\n"
  "}\n"
  "\n"
  "\n"
  "UNITS {\n"
  "	(nA) = (nanoamp)\n"
  "	(mV) = (millivolt)\n"
  "	(umho) = (micromho)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	tau=.25 (ms)\n"
  "	stim=50 (umho)\n"
  "	e=0	(mV)\n"
  "	v	(mV)\n"
  "	gmax=0.1\n"
  "	ntimes=0\n"
  "	\n"
  "	\n"
  "	F0=1\n"
  "	Fmag=0.1\n"
  "	\n"
  "	D0=1\n"
  "	Dmag=0.1\n"
  "	\n"
  "	tauF=94		: from Varela et al.\n"
  "	tauD1=280	: original value=380\n"
  "	tauD2=9200\n"
  "	\n"
  "	from=-1\n"
  "	compart=0	: do not change. assigned dynamically\n"
  "	distance=0	: do not change. assigned dynamically\n"
  "	synlocation=0.5	: do not change. assigned dynamically\n"
  "	\n"
  "	\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	\n"
  "	index\n"
  "	i (nA)\n"
  "	bath (umho)\n"
  "	k (/ms)\n"
  "	onset[SIZE] (ms)\n"
  "	Ridx\n"
  "	treleased[SIZE]\n"
  "	fromlist[SIZE]\n"
  "	NReleased\n"
  "	p\n"
  "	p_release\n"
  "	F\n"
  "	D\n"
  "	\n"
  "	\n"
  "}\n"
  "\n"
  "STATE {\n"
  "	A (umho)\n"
  "	G (umho)\n"
  "\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	k = 1/tau\n"
  "	A = 0\n"
  "	G = 0\n"
  "	index=0\n"
  "	Ridx=0\n"
  "	NReleased=0\n"
  "	p=0\n"
  "	F=F0\n"
  "	D=D0\n"
  "	p_release=0\n"
  "	initvars()\n"
  "	index=0\n"
  "	ntimes=0\n"
  "}\n"
  "\n"
  "PROCEDURE initvars(){\n"
  "	LOCAL i\n"
  "	i=0\n"
  "while(i<SIZE){\n"
  "		onset[i]=1e20\n"
  "		treleased[i]=0\n"
  "		fromlist[i]=-1\n"
  "		i=i+1\n"
  "	}\n"
  "\n"
  "}\n"
  "? current\n"
  "BREAKPOINT {\n"
  "	SOLVE conductance\n"
  "	i = gmax*G*(v - e)\n"
  "}\n"
  "\n"
  ": at each onset time a fixed quantity of material is added to state A\n"
  ": this material moves through G with the form of an alpha function\n"
  "\n"
  "PROCEDURE conductance() { \n"
  "	LOCAL ii,j\n"
  "	while(index < SIZE && t>onset[index]) {\n"
  "		\n"
  "		F=F0\n"
  "		ii=0\n"
  "		while(ii<index){\n"
  "			F = F + Fmag*exp(-(t - onset[ii])/tauF)\n"
  "			ii=ii+1\n"
  "		}\n"
  "		\n"
  "		D=D0\n"
  "		ii=0\n"
  "		while(ii<Ridx){\n"
  "			D = D - Dmag*exp(-(t - treleased[ii])/tauD1)	: use fast depression (tauD1)\n"
  "			ii=ii+1\n"
  "		}\n"
  "		\n"
  "		if(D<0){\n"
  "			D=0\n"
  "		}\n"
  "		\n"
  "		p_release=1-exp(-F*D)\n"
  "		p = scop_random(0)		: uniform distribution between 0 and 1\n"
  "		\n"
  "		if(p<p_release){		: probabilistic release\n"
  "		\n"
  "			from=fromlist[index]\n"
  "			VERBATIM\n"
  "	  	fprintf(stderr,\"\\n Realeased from %.0f onto %.0f at t=%.2f with weight %f\",from, compart, t, gmax);\n"
  "	  	ENDVERBATIM\n"
  "	  \n"
  "			treleased[Ridx]=onset[index]\n"
  "			Ridx=Ridx+1\n"
  "			NReleased=NReleased+1\n"
  "			A = stim		: unitary release\n"
  "		}else{\n"
  "			A=0\n"
  "		}\n"
  "		\n"
  "		index=index+1\n"
  "	}\n"
  "	SOLVE state METHOD sparse\n"
  "	\n"
  "	VERBATIM\n"
  "	return 0;\n"
  "	ENDVERBATIM\n"
  "}\n"
  "\n"
  "? kinetics\n"
  "KINETIC state {\n"
  "	~ A <-> G	(k, 0)\n"
  "	~ G <-> bath	(k, 0)\n"
  "}\n"
  "\n"
  "\n"
  "COMMENT\n"
  "\n"
  "NET_RECEIVE(weight (uS)) {\n"
  "	state_discontinuity(distance, weight)\n"
  "	\n"
  "	if(index<SIZE-1){\n"
  "		onset[index]=t\n"
  "		onset[index+1]=1e20\n"
  "	}\n"
  "	ntimes=ntimes+1\n"
  "	\n"
  "	\n"
  "	VERBATIM\n"
  "	fprintf(stderr,\"\\n %f Realeased onto %f at t=%f\", distance, compart, t);\n"
  "	ENDVERBATIM\n"
  "}\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  ;
#endif
