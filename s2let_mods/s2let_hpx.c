// S2LET package
// Copyright (C) 2012
// Boris Leistedt & Jason McEwen

#include "s2let.h"
#include <stdlib.h>
#include <complex.h>

// Fortran interfaces to Healpix F90 library ; see s2let_hpx.f90
extern void healpix_inverse_real_();
extern void healpix_forward_real_();
extern void healpix_inverse_spin_real_();
extern void healpix_forward_spin_real_();
extern void write_healpix_map_();
extern void read_healpix_map_();
extern void read_healpix_maps_();
extern void healpix_forward_real_();

void s2let_hpx_alm2map_real(double* f, const complex double* flm, int nside, int L)
{
  healpix_inverse_real_(f, flm, &nside, &L);
}

void s2let_hpx_map2alm_real(complex double* flm, const double* f, int nside, int L)
{
  healpix_forward_real_(flm, f, &nside, &L);
}

void s2let_hpx_alm2map_spin_real(double* fQ, double* fU, const complex double* flmE, const complex double* flmB, int nside, int L, int spin)
{
  healpix_inverse_spin_real_(fQ, fU, flmE, flmB, &nside, &L, &spin);
}

void s2let_hpx_map2alm_spin_real(complex double* flmE, complex double* flmB, const double* fQ, const double* fU, int nside, int L, int spin)
{
  healpix_forward_spin_real_(flmE, flmB, fQ, fU, &nside, &L, &spin);
}

void s2let_hpx_read_maps(double* f, char* file, int nside, int nmaps)
{
  read_healpix_maps_(f, file, &nside, &nmaps);
}

void s2let_hpx_read_map(double* f, char* file, int nside)
{
  read_healpix_map_(f, file, &nside);
}

void s2let_hpx_write_map(char* file, const double* f, int nside)
{
  write_healpix_map_(file, f, &nside);
}

void s2let_hpx_allocate_real(double **f, int nside)
{
  *f = calloc(12*nside*nside, sizeof **f);
}

int global_samp = 0;

void s2let_set_hpx_sampling_scheme(int samp)
{
	global_samp = samp;
}

int s2let_scaling_nside(int j, int B, int nside)
{
	if (global_samp == 0)
	{
		printf("B %i\n",B);
		printf("j %i\n",j);
		double n_nside = pow(2,ceil(log(MIN(pow(B,j)/2.0,nside*1.0))/log(2.0)));
		n_nside = pow(2,ceil(log(MIN(pow(B,j)/2,nside*1.0))/log(2.0)));
		n_nside = MAX(n_nside,32);
		printf("- Picking nside = %f\n",n_nside);
		printf("- Picking nside = %i\n",(int)n_nside);
		return (int) n_nside;
	}
	else if (global_samp == 1)
	{
		return nside;
	}
	else if (global_samp == 2)
	{
		return (int) MAX(MAX(pow(2,j)*8,1),nside);
	}
	printf("No valid sampling scheme chosen. Defaulting to nside\n");
	return nside;
}

void s2let_transform_axisym_allocate_hpx_f_wav_hpx_real(double **f_wav, double **f_scal, int nside, const s2let_parameters_t *parameters)
{
	int J_min = parameters->J_min;

	int J = s2let_j_max(parameters);
	int B = (int) parameters->B;
	int j,total = 0;
	for (j=J_min;j<=J;j++)
	{
		total += 12*pow(s2let_scaling_nside(j+1,B,nside),2);
	}
	*f_wav = (double*) calloc(total,sizeof(double));
	*f_scal = (double*) calloc(12*pow(s2let_scaling_nside(J_min,B,nside),2),sizeof(double));
}
