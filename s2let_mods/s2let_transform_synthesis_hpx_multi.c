// S2LET package modificatoin
// Allows multiscale subsampling with HEALPIX
// Sebastian Wagner-Carena

#include "s2let.h"
#include <complex.h>
#include <stdlib.h>

void s2let_transform_axisym_wav_synthesis_hpx_multi(double *f, const double *f_wav, const double *f_scal, int nside, const s2let_parameters_t *parameters)
{
	int L = parameters->L;
	int J_min = parameters->J_min;

	int bandlimit, j, offset, offset_lm;
	int J = s2let_j_max(parameters);

	double *wav_lm, *scal_lm;
	s2let_transform_axisym_lm_allocate_wav(&wav_lm, &scal_lm, parameters);
	s2let_transform_axisym_lm_wav(wav_lm, scal_lm, parameters);

	complex double *flm, *f_wav_lm, *f_scal_lm;
	flm = (complex double*)calloc(L * L, sizeof(complex double));
	s2let_transform_axisym_lm_allocate_f_wav_multires(&f_wav_lm, &f_scal_lm, parameters);

	bandlimit = MIN(s2let_bandlimit(J_min-1, parameters), L);
	s2let_hpx_map2alm_real(f_scal_lm, f_scal, s2let_scaling_nside(J_min-1,nside), bandlimit);
	offset = 0;
	offset_lm = 0;
	for(j = J_min; j <= J; j++){
		bandlimit = MIN(s2let_bandlimit(j, parameters), L);
		int j_nside = s2let_scaling_nside(j,nside);
		s2let_hpx_map2alm_real(f_wav_lm + offset_lm, f_wav + offset, j_nside, bandlimit);
		offset_lm += bandlimit * bandlimit;
		offset += 12 * j_nside * j_nside;
	}

	s2let_transform_axisym_lm_wav_synthesis_multires(flm, f_wav_lm, f_scal_lm, wav_lm, scal_lm, parameters);

	s2let_hpx_alm2map_real(f, flm, nside, L);

	free(flm);
	free(f_scal_lm);
	free(f_wav_lm);
}

/*!
 * PROGRAM : s2let_transform_synthesis_hpx_multi
 * COMMAND : bin/s2let_extract_wav_coeff file B J_min L n_side out_file [samp]
 * ARGUMENTS :
 * - file : fileroot for input healpix maps of wavelet coefficients
 * - B : wavelet parameter
 * - J_min : first wavelet scale to use
 * - L : bandlimit for the decomposition
 * - n_side: the n_side for the original map
 * - out_file : file to which reconstruction will be written
 * - samp : an optional sixth parameter that decides the sampling scheme that
 *		will be used for the wavelet maps. Must be the same as that used for map
 *		generation.
 * OUTPUT : csv files containing the wavelet coefficients are written.
 */
int main(int argc, char *argv[])
{
	printf("--------------------------------------------------\n");
	printf("S2LET library : axisymmetric wavelet transform\n");
	printf("Real signal, HEALPIX sampling\n");
	printf("--------------------------------------------------\n");

	char fileroot[100];
	char outfile[100];
	int L, B, J_min, nside;
	int samp = 0;
	if (sscanf(argv[1], "%s", fileroot) != 1)
		exit(-2);
	if (sscanf(argv[2], "%i", &B) != 1)
		exit(-2);
	if (sscanf(argv[3], "%i", &J_min) != 1)
		exit(-2);
	if (sscanf(argv[4], "%i", &L) != 1)
		exit(-2);
	if (sscanf(argv[5], "%i", &nside) != 1)
		exit(-2);
	if (sscanf(argv[6], "%s", outfile) != 1)
		exit(-2);
	if (argc > 7 && sscanf(argv[7], "%i", &samp) != 1)
		exit(-2);

	s2let_set_hpx_sampling_scheme(samp);

	s2let_parameters_t parameters = {};
	parameters.B = B;
	parameters.L = L;
	parameters.J_min = J_min;

	printf("Parameters for wavelet transform :\n");
	printf("- Wavelet parameter : %i\n", L);
	int J = s2let_j_max(&parameters);
	printf("- Wavelet parameter : %i\n", B);
	printf("- Total number of wavelets : %i\n", J);
	printf("- First wavelet scale to be used : %i\n", J_min);

	char params[1000];
	char file[1000];
	sprintf(params, "%d%s%d%s%d", L, "_", B, "_", J_min);
	int j, offset = 0;
	printf("File root = %s\n",fileroot);

	sprintf(file, "%s%s%s%s", fileroot, "_scal_", params, ".fits");
	printf("- Infile_scal = %s\n",file);
	s2let_fits_hpx_read_nside(file);
	// Allocate memory for wavelets
	double *f_wav, *f_scal;
	s2let_transform_axisym_allocate_hpx_f_wav_hpx_real(&f_wav, &f_scal, nside, &parameters);
	// Read the scaling function
	s2let_hpx_read_map(f_scal, file, s2let_scaling_nside(J_min-1,nside)); // Now write the map to fits file
	// Read the wavelets
	for(j = J_min; j <= J; j++){
		sprintf(file, "%s%s%s%s%d%s", fileroot, "_wav_", params, "_", j, ".fits");
		printf("- Infile_wav[j=%i] = %s\n",j,file);
		int j_nside = s2let_scaling_nside(j,nside);
		s2let_hpx_read_map(f_wav + offset, file, j_nside); // Now write the map to fits file
		offset += 12*j_nside*j_nside; // Go to the next wavelet
	}

	// Allocate memory for reconstruction
	double *f = (double*)calloc(12*nside*nside, sizeof(double));
	printf("File successfully read from file\n");

	printf("Performing wavelet reconstruction...");fflush(NULL);
	s2let_transform_axisym_wav_synthesis_hpx_multi(f, f_wav, f_scal, nside, &parameters);
	printf("done\n");

	// Output the wavelets to FITS files
	printf("Writing reconsturcted map to FITS files\n");
	printf("- Outfile = %s\n",outfile);
	remove(outfile); // In case the file exists
	s2let_hpx_write_map(outfile, f, nside); // Now write the map to fits file

	printf("--------------------------------------------------\n");


}