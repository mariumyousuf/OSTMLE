NEURON {     
        SUFFIX rchan
        NONSPECIFIC_CURRENT i
	RANGE noise,gnoise,nevents
        RANGE mean, stdev
}

PARAMETER {
        mean = 0
        stdev = .001
	nevents=1
	gnoise=0	(mho/cm2)
	noise=0	 
}

ASSIGNED {
        i (milliamp/cm2)
        i_ran (milliamp/cm2)
}

BREAKPOINT {
        SOLVE ran
	if(noise>nevents){
        	i = i_ran - gnoise*noise
	}else{
		i=i_ran
	}
}

PROCEDURE ran() {
        i_ran = normrand(-mean, stdev)
}

