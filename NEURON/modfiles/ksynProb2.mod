COMMENT
alpha function synapse implemented as continuously integrated
kinetic scheme a la Srinivasan and Chiel (Neural Computation) so that
one can give many stimuli which summate.

Onset times are placed in the vector onset[SIZE]
Conductance located in state variable G
The amplitude of each individual alpha function is given by stim,
stim * t * exp(-t/tau).
The last onset time should be a very large number so stim stops getting
added to state A

Jean-Marc: 
-Included probabilistic release, facilitation and short-term depression, a la Maas&Zador.
-Included presynaptic NetCon trigger

ENDCOMMENT

DEFINE SIZE 2000

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}


NEURON {
	POINT_PROCESS KSynProb
	RANGE tau, stim, e, i,onset, treleased, fromlist, from, gmax, p,p_release,compart,ntimes,distance,synlocation,D0,F0,Dmag,Fmag, Ridx, NReleased
	NONSPECIFIC_CURRENT i
}


UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(umho) = (micromho)
}

PARAMETER {
	tau=.25 (ms)
	stim=50 (umho)
	e=0	(mV)
	v	(mV)
	gmax=0.1
	ntimes=0
	
	
	F0=1
	Fmag=0.1
	
	D0=1
	Dmag=0.1
	
	tauF=94		: from Varela et al.
	tauD1=280	: original value=380
	tauD2=9200
	
	from=-1
	compart=0	: do not change. assigned dynamically
	distance=0	: do not change. assigned dynamically
	synlocation=0.5	: do not change. assigned dynamically
	
	
}

ASSIGNED {
	
	index
	i (nA)
	bath (umho)
	k (/ms)
	onset[SIZE] (ms)
	Ridx
	treleased[SIZE]
	fromlist[SIZE]
	NReleased
	p
	p_release
	F
	D
	
	
}

STATE {
	A (umho)
	G (umho)

}

INITIAL {
	k = 1/tau
	A = 0
	G = 0
	index=0
	Ridx=0
	NReleased=0
	p=0
	F=F0
	D=D0
	p_release=0
	initvars()
	index=0
	ntimes=0
}

PROCEDURE initvars(){
	LOCAL i
	i=0
while(i<SIZE){
		onset[i]=1e20
		treleased[i]=0
		fromlist[i]=-1
		i=i+1
	}

}
? current
BREAKPOINT {
	SOLVE conductance
	i = gmax*G*(v - e)
}

: at each onset time a fixed quantity of material is added to state A
: this material moves through G with the form of an alpha function

PROCEDURE conductance() { 
	LOCAL ii,j
	while(index < SIZE && t>onset[index]) {
		
		F=F0
		ii=0
		while(ii<index){
			F = F + Fmag*exp(-(t - onset[ii])/tauF)
			ii=ii+1
		}
		
		D=D0
		ii=0
		while(ii<Ridx){
			D = D - Dmag*exp(-(t - treleased[ii])/tauD1)	: use fast depression (tauD1)
			ii=ii+1
		}
		
		if(D<0){
			D=0
		}
		
		p_release=1-exp(-F*D)
		p = scop_random(0)		: uniform distribution between 0 and 1
		
		if(p<p_release){		: probabilistic release
		
			from=fromlist[index]
			VERBATIM
	  	fprintf(stderr,"\n Realeased from %.0f onto %.0f at t=%.2f with weight %f",from, compart, t, gmax);
	  	ENDVERBATIM
	  
			treleased[Ridx]=onset[index]
			Ridx=Ridx+1
			NReleased=NReleased+1
			A = stim		: unitary release
		}else{
			A=0
		}
		
		index=index+1
	}
	SOLVE state METHOD sparse
	
	VERBATIM
	return 0;
	ENDVERBATIM
}

? kinetics
KINETIC state {
	~ A <-> G	(k, 0)
	~ G <-> bath	(k, 0)
}


COMMENT

NET_RECEIVE(weight (uS)) {
	state_discontinuity(distance, weight)
	
	if(index<SIZE-1){
		onset[index]=t
		onset[index+1]=1e20
	}
	ntimes=ntimes+1
	
	
	VERBATIM
	fprintf(stderr,"\n %f Realeased onto %f at t=%f", distance, compart, t);
	ENDVERBATIM
}

ENDCOMMENT

