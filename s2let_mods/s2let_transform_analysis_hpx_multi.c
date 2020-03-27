// S2LET package modificatoin
// Allows multiscale subsampling with HEALPIX
// Sebastian Wagner-Carena

#include "s2let.h"
#include <complex.h>
#include <stdlib.h>

void s2let_transform_axisym_wav_analysis_hpx_multi(double *f_wav, double *f_scal, const double *f, int nside, const s2let_parameters_t *parameters)
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

	s2let_hpx_map2alm_real(flm, f, nside, L);

	s2let_transform_axisym_lm_wav_analysis_multires(f_wav_lm, f_scal_lm, flm, wav_lm, scal_lm, parameters);

	bandlimit = MIN(s2let_bandlimit(J_min-1, parameters), L);
	s2let_hpx_alm2map_real(f_scal, f_scal_lm, s2let_scaling_nside(J_min-1,nside), bandlimit);

	offset = 0;
	offset_lm = 0;
	for(j = J_min; j <= J; j++){
		bandlimit = MIN(s2let_bandlimit(j, parameters), L);
		int j_nside = s2let_scaling_nside(j,nside);
		s2let_hpx_alm2map_real(f_wav + offset, f_wav_lm + offset_lm, j_nside, bandlimit);
		offset_lm += bandlimit * bandlimit;
		offset += 12 * j_nside * j_nside;
	}

	free(flm);
	free(f_scal_lm);
	free(f_wav_lm);
}

/*!
 * PROGRAM : s2let_transform_analysis_hpx_multi
 * COMMAND : bin/s2let_transform_axisym_hpx_multi file B J_min L out_file [samp]
 * ARGUMENTS :
 * - file : input healpix map
 * - B : wavelet parameter
 * - J_min : first wavelet scale to use
 * - L : bandlimit for the decomposition
 * - out_file : prefix for output wavelet coefficient fits files 
 * - samp : an optional sixth parameter that decides the sampling scheme that
 *		will be used for the wavelet maps. 0 is minimal sampling, 1 is full
 *		resolution sampling, and 2 is oversampling (wavelet maps will have larger
 *		nside than the original map).
 * OUTPUT : fits files containing the wavelet coefficients are written.
 */
int main(int argc, char *argv[])
{
	char file[100];
	char out_file_prefix[100];
	if (sscanf(argv[1], "%s", file) != 1)
		exit(-2);
	printf("Input HEALPIX map : %s\n",file);
	const int nside = s2let_fits_hpx_read_nside(file);
	printf("- Detected bandlimit nside = %i\n",nside);
	int L, J_min;
	int samp = 0;
	int B;
	if (sscanf(argv[2], "%i", &B) != 1)
		exit(-2);
	if (sscanf(argv[3], "%i", &J_min) != 1)
		exit(-2);
	if (sscanf(argv[4], "%i", &L) != 1)
		exit(-2);
	if (sscanf(argv[5], "%s", out_file_prefix) != 1)
		exit(-2);
	if (argc > 6 && sscanf(argv[6], "%i", &samp) != 1)
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

	// Read MW map from file
	double *f = (double*)calloc(12*nside*nside, sizeof(double));
	s2let_hpx_read_map(f, file, nside);
	printf("File successfully read from file\n");

	printf("Performing wavelet decomposition...");fflush(NULL);
	double *f_wav, *f_scal;
	s2let_transform_axisym_allocate_hpx_f_wav_hpx_real(&f_wav, &f_scal, nside, 
		&parameters);
	s2let_transform_axisym_wav_analysis_hpx_multi(f_wav, f_scal, f, nside, 
		&parameters);
	printf("done\n");

	// Output the wavelets to FITS files
	printf("Writing wavelet maps to FITS files\n");
	char outfile[1000];
	char params[1000];
	sprintf(params, "%d%s%d%s%d", L, "_", B, "_", J_min);
	int j; // Explicitly compute the maximum wavelet scale
	int offset = 0; // Start with the first wavelet

	for(j = J_min; j <= J; j++){
		sprintf(outfile, "%s%s%s%s%d%s", out_file_prefix, "_wav_", params, "_", j, ".fits");
		printf("- Outfile_wav[j=%i] = %s\n",j,outfile);
		int j_nside = s2let_scaling_nside(j,nside);
		remove(outfile); // In case the file exists
		s2let_hpx_write_map(outfile, f_wav + offset, j_nside); // Now write the map to fits file
		offset += 12*j_nside*j_nside; // Go to the next wavelet
	}
	// Finally write the scaling function
	sprintf(outfile, "%s%s%s%s", out_file_prefix, "_scal_", params, ".fits");
	printf("- Outfile_scal = %s\n",outfile);
	remove(outfile); // In case the file exists
	s2let_hpx_write_map(outfile, f_scal, s2let_scaling_nside(J_min-1,nside)); // Now write the map to fits file

	printf("--------------------------------------------------\n");


}