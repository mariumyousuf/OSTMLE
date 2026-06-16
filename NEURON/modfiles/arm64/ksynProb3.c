/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
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
 
#define nrn_init _nrn_init__KSynProbFull
#define _nrn_initial _nrn_initial__KSynProbFull
#define nrn_cur _nrn_cur__KSynProbFull
#define _nrn_current _nrn_current__KSynProbFull
#define nrn_jacob _nrn_jacob__KSynProbFull
#define nrn_state _nrn_state__KSynProbFull
#define _net_receive _net_receive__KSynProbFull 
#define _f_rates _f_rates__KSynProbFull 
#define conductance conductance__KSynProbFull 
#define initvars initvars__KSynProbFull 
#define rates rates__KSynProbFull 
#define state state__KSynProbFull 
 
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
#define tau_columnindex 0
#define stim _p[1]
#define stim_columnindex 1
#define e _p[2]
#define e_columnindex 2
#define gmax _p[3]
#define gmax_columnindex 3
#define ntimes _p[4]
#define ntimes_columnindex 4
#define F0 _p[5]
#define F0_columnindex 5
#define Fmag _p[6]
#define Fmag_columnindex 6
#define D0 _p[7]
#define D0_columnindex 7
#define Dmag _p[8]
#define Dmag_columnindex 8
#define from _p[9]
#define from_columnindex 9
#define compart _p[10]
#define compart_columnindex 10
#define distance _p[11]
#define distance_columnindex 11
#define synlocation _p[12]
#define synlocation_columnindex 12
#define gmax_nmda _p[13]
#define gmax_nmda_columnindex 13
#define i _p[14]
#define i_columnindex 14
#define onset (_p + 15)
#define onset_columnindex 15
#define Ridx _p[5015]
#define Ridx_columnindex 5015
#define treleased (_p + 5016)
#define treleased_columnindex 5016
#define fromlist (_p + 10016)
#define fromlist_columnindex 10016
#define NReleased _p[15016]
#define NReleased_columnindex 15016
#define p _p[15017]
#define p_columnindex 15017
#define p_release _p[15018]
#define p_release_columnindex 15018
#define gNMDA _p[15019]
#define gNMDA_columnindex 15019
#define C _p[15020]
#define C_columnindex 15020
#define A _p[15021]
#define A_columnindex 15021
#define G _p[15022]
#define G_columnindex 15022
#define C0 _p[15023]
#define C0_columnindex 15023
#define C1 _p[15024]
#define C1_columnindex 15024
#define C2 _p[15025]
#define C2_columnindex 15025
#define Ds _p[15026]
#define Ds_columnindex 15026
#define O _p[15027]
#define O_columnindex 15027
#define B _p[15028]
#define B_columnindex 15028
#define index _p[15029]
#define index_columnindex 15029
#define bath _p[15030]
#define bath_columnindex 15030
#define k _p[15031]
#define k_columnindex 15031
#define F _p[15032]
#define F_columnindex 15032
#define D _p[15033]
#define D_columnindex 15033
#define DA _p[15034]
#define DA_columnindex 15034
#define DG _p[15035]
#define DG_columnindex 15035
#define DC0 _p[15036]
#define DC0_columnindex 15036
#define DC1 _p[15037]
#define DC1_columnindex 15037
#define DC2 _p[15038]
#define DC2_columnindex 15038
#define DDs _p[15039]
#define DDs_columnindex 15039
#define DO _p[15040]
#define DO_columnindex 15040
#define DB _p[15041]
#define DB_columnindex 15041
#define _g _p[15042]
#define _g_columnindex 15042
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
 static double _hoc_rates(void*);
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
 "rates", _hoc_rates,
 0, 0
};
 /* declare global and static user variables */
#define Rc Rc_KSynProbFull
 double Rc = 0.0738;
#define Ro Ro_KSynProbFull
 double Ro = 0.0465;
#define Rr Rr_KSynProbFull
 double Rr = 0.0068;
#define Rd Rd_KSynProbFull
 double Rd = 0.0084;
#define Ru Ru_KSynProbFull
 double Ru = 0.0129;
#define Rb Rb_KSynProbFull
 double Rb = 0.005;
#define mg mg_KSynProbFull
 double mg = 1;
#define rb rb_KSynProbFull
 double rb = 0;
#define tauD2 tauD2_KSynProbFull
 double tauD2 = 9200;
#define tauD1 tauD1_KSynProbFull
 double tauD1 = 400;
#define tauF tauF_KSynProbFull
 double tauF = 94;
#define usetable usetable_KSynProbFull
 double usetable = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "usetable_KSynProbFull", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "Rb_KSynProbFull", "/uM",
 "Ru_KSynProbFull", "/ms",
 "Rd_KSynProbFull", "/ms",
 "Rr_KSynProbFull", "/ms",
 "Ro_KSynProbFull", "/ms",
 "Rc_KSynProbFull", "/ms",
 "mg_KSynProbFull", "mM",
 "rb_KSynProbFull", "/ms",
 "tau", "ms",
 "stim", "umho",
 "e", "mV",
 "gmax_nmda", "umho",
 "A", "umho",
 "G", "umho",
 "i", "nA",
 "onset", "ms",
 "gNMDA", "umho",
 "C", "mM",
 0,0
};
 static double A0 = 0;
 static double B0 = 0;
 static double C20 = 0;
 static double C10 = 0;
 static double C00 = 0;
 static double Ds0 = 0;
 static double G0 = 0;
 static double O0 = 0;
 static double delta_t = 1;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "tauF_KSynProbFull", &tauF_KSynProbFull,
 "tauD1_KSynProbFull", &tauD1_KSynProbFull,
 "tauD2_KSynProbFull", &tauD2_KSynProbFull,
 "Rb_KSynProbFull", &Rb_KSynProbFull,
 "Ru_KSynProbFull", &Ru_KSynProbFull,
 "Rd_KSynProbFull", &Rd_KSynProbFull,
 "Rr_KSynProbFull", &Rr_KSynProbFull,
 "Ro_KSynProbFull", &Ro_KSynProbFull,
 "Rc_KSynProbFull", &Rc_KSynProbFull,
 "mg_KSynProbFull", &mg_KSynProbFull,
 "rb_KSynProbFull", &rb_KSynProbFull,
 "usetable_KSynProbFull", &usetable_KSynProbFull,
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
"KSynProbFull",
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
 "gmax_nmda",
 0,
 "i",
 "onset[5000]",
 "Ridx",
 "treleased[5000]",
 "fromlist[5000]",
 "NReleased",
 "p",
 "p_release",
 "gNMDA",
 "C",
 0,
 "A",
 "G",
 "C0",
 "C1",
 "C2",
 "Ds",
 "O",
 "B",
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
 	_p = nrn_prop_data_alloc(_mechtype, 15043, _prop);
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
 	gmax_nmda = 0.001;
  }
 	_prop->param = _p;
 	_prop->param_size = 15043;
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

 void _ksynProb3_reg() {
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
  hoc_register_prop_size(_mechtype, 15043, 2);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
 	hoc_register_cvode(_mechtype, _ode_count, 0, 0, 0);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 KSynProbFull /Users/mariumyousuf/Desktop/causal/fall2025/NEURONFA25/CoreSmallNet/modfiles/ksynProb3.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double *_t_B;
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int _f_rates(double);
static int conductance();
static int initvars();
static int rates(double);
 extern double *_getelm();
 
#define _MATELM1(_row,_col)	*(_getelm(_row + 1, _col + 1))
 
#define _RHS1(_arg) _coef1[_arg + 1]
 static double *_coef1;
 
#define _linmat1  1
 static void* _sparseobj1;
 static void _n_rates(double);
 static int _slist1[7], _dlist1[7]; static double *_temp1;
 static int state();
 
static int  initvars (  ) {
   double _li ;
 _li = 0.0 ;
   while ( _li < 5000.0 ) {
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
   double _lii ;
 C = 0.0 ;
   while ( index < 5000.0  && t > onset [ ((int) index ) ] ) {
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
//	  		fprintf(stderr,"\n Realeased from %.0f onto %.0f at t=%.2f with weight %f",from, compart, t, gmax);
 treleased [ ((int) Ridx ) ] = onset [ ((int) index ) ] ;
       Ridx = Ridx + 1.0 ;
       NReleased = NReleased + 1.0 ;
       A = stim ;
       C = 70.0 ;
       }
     else {
       A = 0.0 ;
       C = 0.0 ;
       }
     index = index + 1.0 ;
     }
   error = sparse(&_sparseobj1, 7, _slist1, _dlist1, _p, &t, dt, state,&_coef1, _linmat1);
 if(error){fprintf(stderr,"at line 256 in file ksynProb3.mod:\n	\n"); nrn_complain(_p); abort_run(error);}
    if (secondorder) {
    int _i;
    for (_i = 0; _i < 7; ++_i) {
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
for(_i=1;_i<7;_i++){
  	_RHS1(_i) = -_dt1*(_p[_slist1[_i]] - _p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
} }
 /* ~ A <-> G ( k , 0.0 )*/
 f_flux =  k * A ;
 b_flux =  0.0 * G ;
 _RHS1( 1) -= (f_flux - b_flux);
 _RHS1( 6) += (f_flux - b_flux);
 
 _term =  k ;
 _MATELM1( 1 ,1)  += _term;
 _MATELM1( 6 ,1)  -= _term;
 _term =  0.0 ;
 _MATELM1( 1 ,6)  -= _term;
 _MATELM1( 6 ,6)  += _term;
 /*REACTION*/
  /* ~ G <-> bath ( k , 0.0 )*/
 f_flux =  k * G ;
 b_flux =  0.0 * bath ;
 _RHS1( 6) -= (f_flux - b_flux);
 
 _term =  k ;
 _MATELM1( 6 ,6)  += _term;
 /*REACTION*/
  rb = Rb * ( 1e3 ) * C ;
   /* ~ C0 <-> C1 ( rb , Ru )*/
 f_flux =  rb * C0 ;
 b_flux =  Ru * C1 ;
 _RHS1( 4) -= (f_flux - b_flux);
 _RHS1( 3) += (f_flux - b_flux);
 
 _term =  rb ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 3 ,4)  -= _term;
 _term =  Ru ;
 _MATELM1( 4 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
  /* ~ C1 <-> C2 ( rb , Ru )*/
 f_flux =  rb * C1 ;
 b_flux =  Ru * C2 ;
 _RHS1( 3) -= (f_flux - b_flux);
 _RHS1( 2) += (f_flux - b_flux);
 
 _term =  rb ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 2 ,3)  -= _term;
 _term =  Ru ;
 _MATELM1( 3 ,2)  -= _term;
 _MATELM1( 2 ,2)  += _term;
 /*REACTION*/
  /* ~ C2 <-> Ds ( Rd , Rr )*/
 f_flux =  Rd * C2 ;
 b_flux =  Rr * Ds ;
 _RHS1( 2) -= (f_flux - b_flux);
 _RHS1( 5) += (f_flux - b_flux);
 
 _term =  Rd ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 5 ,2)  -= _term;
 _term =  Rr ;
 _MATELM1( 2 ,5)  -= _term;
 _MATELM1( 5 ,5)  += _term;
 /*REACTION*/
  /* ~ C2 <-> O ( Ro , Rc )*/
 f_flux =  Ro * C2 ;
 b_flux =  Rc * O ;
 _RHS1( 2) -= (f_flux - b_flux);
 
 _term =  Ro ;
 _MATELM1( 2 ,2)  += _term;
 _term =  Rc ;
 _MATELM1( 2 ,0)  -= _term;
 /*REACTION*/
   /* C0 + C1 + C2 + Ds + O = 1.0 */
 _RHS1(0) =  1.0;
 _MATELM1(0, 0) = 1;
 _RHS1(0) -= O ;
 _MATELM1(0, 5) = 1;
 _RHS1(0) -= Ds ;
 _MATELM1(0, 2) = 1;
 _RHS1(0) -= C2 ;
 _MATELM1(0, 3) = 1;
 _RHS1(0) -= C1 ;
 _MATELM1(0, 4) = 1;
 _RHS1(0) -= C0 ;
 /*CONSERVATION*/
   } return _reset;
 }
 static double _mfac_rates, _tmin_rates;
 static void _check_rates();
 static void _check_rates() {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  static double _sav_mg;
  if (!usetable) {return;}
  if (_sav_mg != mg) { _maktable = 1;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_rates =  - 120.0 ;
   _tmax =  100.0 ;
   _dx = (_tmax - _tmin_rates)/200.; _mfac_rates = 1./_dx;
   for (_i=0, _x=_tmin_rates; _i < 201; _x += _dx, _i++) {
    _f_rates(_x);
    _t_B[_i] = B;
   }
   _sav_mg = mg;
  }
 }

 static int rates(double _lv){ _check_rates();
 _n_rates(_lv);
 return 0;
 }

 static void _n_rates(double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 _f_rates(_lv); return; 
}
 _xi = _mfac_rates * (_lv - _tmin_rates);
 if (isnan(_xi)) {
  B = _xi;
  return;
 }
 if (_xi <= 0.) {
 B = _t_B[0];
 return; }
 if (_xi >= 200.) {
 B = _t_B[200];
 return; }
 _i = (int) _xi;
 _theta = _xi - (double)_i;
 B = _t_B[_i] + _theta*(_t_B[_i+1] - _t_B[_i]);
 }

 
static int  _f_rates (  double _lv ) {
   B = 1.0 / ( 1.0 + exp ( 0.062 * - _lv ) * ( mg / 3.57 ) ) ;
    return 0; }
 
static double _hoc_rates(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
  _r = 1.;
 rates (  *getarg(1) );
 return(_r);
}
 
static int _ode_count(int _type){ hoc_execerror("KSynProbFull", "cannot be used with CVODE"); return 0;}

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  A = A0;
  B = B0;
  C2 = C20;
  C1 = C10;
  C0 = C00;
  Ds = Ds0;
  G = G0;
  O = O0;
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
   rates ( _threadargscomma_ v ) ;
   C0 = 1.0 ;
   C = 0.0 ;
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
   rates ( _threadargscomma_ v ) ;
   gNMDA = gmax_nmda * O * B ;
   i = gmax * G * ( v - e ) + gNMDA * ( v - e ) ;
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
 if(error){fprintf(stderr,"at line 200 in file ksynProb3.mod:\n	\n"); nrn_complain(_p); abort_run(error);}
 }}}
 dt = _dtsav;
}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
   _t_B = makevector(201*sizeof(double));
 _slist1[0] = O_columnindex;  _dlist1[0] = DO_columnindex;
 _slist1[1] = A_columnindex;  _dlist1[1] = DA_columnindex;
 _slist1[2] = C2_columnindex;  _dlist1[2] = DC2_columnindex;
 _slist1[3] = C1_columnindex;  _dlist1[3] = DC1_columnindex;
 _slist1[4] = C0_columnindex;  _dlist1[4] = DC0_columnindex;
 _slist1[5] = Ds_columnindex;  _dlist1[5] = DDs_columnindex;
 _slist1[6] = G_columnindex;  _dlist1[6] = DG_columnindex;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/mariumyousuf/Desktop/causal/fall2025/NEURONFA25/CoreSmallNet/modfiles/ksynProb3.mod";
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
  "JM: Included probabilistic release, facilitation and 1-2 short-term depression after Varela et al, J neuroscience\n"
  " \n"
  "\n"
  "\n"
  "\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  "DEFINE SIZE 5000\n"
  "\n"
  "INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}\n"
  "\n"
  "\n"
  "NEURON {\n"
  "	POINT_PROCESS KSynProbFull\n"
  "	RANGE tau, stim, e, i,onset, treleased, fromlist,from, gmax,  p,p_release,compart,ntimes,distance,synlocation,D0,F0,Dmag,Fmag, Ridx, NReleased\n"
  "	RANGE C,C0, C1, C2, Ds, O, B\n"
  "	RANGE gNMDA, gmax_nmda\n"
  "	GLOBAL mg, Rb, Ru, Rd, Rr, Ro, Rc,rb\n"
  "	\n"
  "	NONSPECIFIC_CURRENT i\n"
  "}\n"
  "\n"
  "\n"
  "UNITS {\n"
  "	(nA) = (nanoamp)\n"
  "	(mV) = (millivolt)\n"
  "	(pS) = (picosiemens)\n"
  "	(umho) = (micromho)\n"
  "	(mM) = (milli/liter)\n"
  "	(uM) = (micro/liter)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	tau=.25 (ms)\n"
  "	stim=50 (umho)\n"
  "	e=0	(mV)\n"
  "	v	(mV)\n"
  "	gmax=0.1\n"
  "	\n"
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
  "	tauD1=400\n"
  "	tauD2=9200\n"
  "	\n"
  "	from=-1\n"
  "	compart=0	: do not change. assigned dynamically\n"
  "	distance=0	: do not change. assigned dynamically\n"
  "	synlocation=0.5	: do not change. assigned dynamically\n"
  "	\n"
  "	: Destexhe, Mainen & Sejnowski, 1996\n"
  "	Rb	= 5e-3    (/uM /ms)	: binding 		\n"
  "	Ru	= 12.9e-3  (/ms)	: unbinding		\n"
  "	Rd	= 8.4e-3   (/ms)	: desensitization\n"
  "	Rr	= 6.8e-3   (/ms)	: resensitization \n"
  "	Ro	= 46.5e-3   (/ms)	: opening\n"
  "	Rc	= 73.8e-3   (/ms)	: closing\n"
  "	\n"
  "	gmax_nmda=0.001 (umho)\n"
  "	mg	= 1    (mM)		: external magnesium concentration\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "COMMENT\n"
  "	: Clements et al. 1992\n"
  "	Rb	= 5e-3    (/uM /ms)	: binding 		\n"
  "	Ru	= 9.5e-3  (/ms)	: unbinding		\n"
  "	Rd	= 16e-3   (/ms)	: desensitization\n"
  "	Rr	= 13e-3   (/ms)	: resensitization \n"
  "	Ro	= 25e-3   (/ms)	: opening\n"
  "	Rc	= 59e-3   (/ms)	: closing\n"
  "\n"
  "	: Hessler Shirke & Malinow 1993\n"
  "	Rb	= 5e-3    (/uM /ms)	: binding 		\n"
  "	Ru	= 9.5e-3  (/ms)	: unbinding		\n"
  "	Rd	= 16e-3   (/ms)	: desensitization\n"
  "	Rr	= 13e-3   (/ms)	: resensitization \n"
  "	Ro	= 25e-3   (/ms)	: opening\n"
  "	Rc	= 59e-3   (/ms)	: closing\n"
  "\n"
  "	: Clements & Westbrook 1991\n"
  "	Rb	=  5    (uM /s)	: binding 		\n"
  "	Ru	=  5	(/s)	: unbinding -> gives Kd = Rb/Ru = 1 uM\n"
  "	Rd	=  4.0  (/s)	: desensitization\n"
  "	Rr	=  0.3  (/s)	: resensitization \n"
  "	Ro	= 10  (/s)	: opening\n"
  "	Rc	= 322   (/s)	: closing\n"
  "\n"
  "	: Edmonds & Colquhoun 1992\n"
  "	Rb	=  5    (uM /s)	: binding 		\n"
  "	Ru	=  4.7  (/s)	: unbinding		\n"
  "	Rd	=  8.4  (/s)	: desensitization\n"
  "	Rr	=  1.8  (/s)	: resensitization \n"
  "	Ro	= 46.5  (/s)	: opening\n"
  "	Rc	= 91.6  (/s)	: closing\n"
  "\n"
  "	: Lester & Jahr 1992\n"
  "	Rb	= 5    (uM /s)	: binding 		\n"
  "	Ru	= 6.7   (/s)	: unbinding		\n"
  "	Rd	= 15.2  (/s)	: desensitization\n"
  "	Rr	= 9.4   (/s)	: resensitization \n"
  "	Ro	= 83.8  (/s)	: opening\n"
  "	Rc	= 83.8  (/s)	: closing\n"
  "\n"
  "ENDCOMMENT\n"
  "ASSIGNED {\n"
  "	\n"
  "	index\n"
  "	i (nA)\n"
  "	bath (umho)\n"
  "	k (/ms)\n"
  "	\n"
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
  "	gNMDA 		(umho)		: conductance\n"
  "	C 		(mM)		: pointer to glutamate concentration\n"
  "	rb		(/ms)    : binding\n"
  "	\n"
  "}\n"
  "\n"
  "STATE {\n"
  "	A (umho)\n"
  "	G (umho)\n"
  "	: NMDA Channel states (all fractions)\n"
  "	\n"
  "	C0		: unbound\n"
  "	C1		: single bound\n"
  "	C2		: double bound\n"
  "	Ds		: desensitized\n"
  "	O		: open\n"
  "\n"
  "	B		: fraction free of Mg2+ block\n"
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
  "	\n"
  "	initvars()\n"
  "	index=0\n"
  "	ntimes=0\n"
  "	\n"
  "	rates(v)\n"
  "	C0 = 1\n"
  "	C=0\n"
  "}\n"
  "\n"
  "PROCEDURE initvars(){\n"
  "	LOCAL i\n"
  "	i=0\n"
  "	while(i<SIZE){\n"
  "		onset[i]=1e20\n"
  "		treleased[i]=0\n"
  "		fromlist[i]=-1\n"
  "		i=i+1\n"
  "	}\n"
  "}\n"
  "\n"
  "\n"
  "? current\n"
  "BREAKPOINT {\n"
  "	rates(v)\n"
  "	SOLVE conductance\n"
  "	\n"
  "	gNMDA = gmax_nmda * O * B\n"
  "\n"
  "	i = gmax*G*(v - e)+  gNMDA * (v - e)\n"
  "}\n"
  "\n"
  ": at each onset time a fixed quantity of material is added to state A\n"
  ": this material moves through G with the form of an alpha function\n"
  "\n"
  "PROCEDURE conductance() { \n"
  "	LOCAL ii\n"
  "	C=0\n"
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
  "//	  		fprintf(stderr,\"\\n Realeased from %.0f onto %.0f at t=%.2f with weight %f\",from, compart, t, gmax);\n"
  "	  		ENDVERBATIM\n"
  "	  	\n"
  "			treleased[Ridx]=onset[index]\n"
  "			Ridx=Ridx+1\n"
  "			NReleased=NReleased+1\n"
  "			A = stim		: unitary release\n"
  "			C=70\n"
  "		}else{\n"
  "			A=0\n"
  "			C=0\n"
  "\n"
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
  "\n"
  "? kinetics\n"
  "KINETIC state {\n"
  "	~ A <-> G	(k, 0)\n"
  "	~ G <-> bath	(k, 0)\n"
  "						:NMDA 5 states\n"
  "	 rb = Rb * (1e3) * C\n"
  "	~ C0 <-> C1	(rb,Ru)\n"
  "	~ C1 <-> C2	(rb,Ru)\n"
  "	~ C2 <-> Ds	(Rd,Rr)\n"
  "	~ C2 <-> O	(Ro,Rc)\n"
  "\n"
  "	CONSERVE C0+C1+C2+Ds+O = 1\n"
  "}\n"
  "\n"
  "PROCEDURE rates(v(mV)) {\n"
  "	TABLE B\n"
  "	DEPEND mg\n"
  "	FROM -120 TO 100 WITH 200\n"
  "\n"
  "	: from Jahr & Stevens\n"
  "\n"
  "	B = 1 / (1 + exp(0.062 (/mV) * -v) * (mg / 3.57 (mM)))\n"
  "}\n"
  "\n"
  ;
#endif
