import numpy as np
import healpy as hp
from hgmca import wavelets
import unittest, os
import scipy.integrate as integrate


class TestBaseFunctions(unittest.TestCase):

	def setUp(self, *args, **kwargs):
		self.root_path = os.path.dirname(os.path.abspath(__file__))+'/test_data/'

	def test_k_sdw(self):
		# Test that the modified Schwartz function is returning the
		# expected values
		scale_int = 2
		self.assertAlmostEqual(wavelets.k_sdw(0.3,scale_int),0)
		self.assertAlmostEqual(wavelets.k_sdw(0.6,scale_int),
			0.073228222705679)
		self.assertAlmostEqual(wavelets.k_sdw(0.9,scale_int),
			0.04881881513711933)
		self.assertAlmostEqual(wavelets.k_sdw(0.95,scale_int),
			0.00406938962049771)

	def test_kappa_integral(self):
		# Test that the integration is equivalent to scipy integration.
		scale_int = 2
		n_quads = 1000
		lower = 0.7
		upper = 1.0

		self.assertAlmostEqual(wavelets.kappa_integral(lower,upper,
			n_quads,scale_int),integrate.quad(wavelets.k_sdw,lower,upper,
			args=(scale_int))[0])

		lower = 0.6
		self.assertAlmostEqual(wavelets.kappa_integral(lower,upper,
			n_quads,scale_int),integrate.quad(wavelets.k_sdw,lower,upper,
			args=(scale_int))[0])

		lower = 0.5
		self.assertAlmostEqual(wavelets.kappa_integral(lower,upper,
			n_quads,scale_int),integrate.quad(wavelets.k_sdw,lower,upper,
			args=(scale_int))[0])

		scale_int = 3
		self.assertAlmostEqual(wavelets.kappa_integral(lower,upper,
			n_quads,scale_int),integrate.quad(wavelets.k_sdw,lower,upper,
			args=(scale_int))[0])

	def test_j_max(self):
		# Test that j_max returns the expected results
		self.assertEqual(wavelets.calc_j_max(128,2),7)
		self.assertEqual(wavelets.calc_j_max(256,2),8)
		self.assertEqual(wavelets.calc_j_max(255,2),8)
		self.assertEqual(wavelets.calc_j_max(2048,2),11)

	def test_phi2_s2dw(self):
		# Test that the phi2_s2dw integral behaves as it should.
		scale_int = 2
		band_lim = 1024
		j_max = wavelets.calc_j_max(band_lim,scale_int)
		phi2 = np.zeros((j_max+2)*band_lim)
		n_quads = 1000
		norm = wavelets.kappa_integral(1.0/scale_int,1.0,n_quads,scale_int)

		wavelets.phi2_s2dw(phi2,band_lim,scale_int,n_quads)
		self.assertAlmostEqual(np.max(phi2),1)
		self.assertEqual(np.min(phi2),0)

		# We'll go an manually test the values for a few scales
		correct_values = np.zeros(band_lim)
		correct_values[0] = 1
		np.testing.assert_almost_equal(phi2[0:band_lim],correct_values)

		correct_values = np.zeros(band_lim)
		correct_values[0:2] = 1
		np.testing.assert_almost_equal(phi2[band_lim:2*band_lim],
			correct_values)

		correct_values = np.zeros(band_lim)
		correct_values[0:2] = 1
		correct_values[2] = wavelets.kappa_integral(2/4,1.0,n_quads,
			scale_int)/norm
		correct_values[3] = wavelets.kappa_integral(3/4,1.0,n_quads,
			scale_int)/norm
		np.testing.assert_almost_equal(phi2[2*band_lim:3*band_lim],
			correct_values)

		correct_values = np.ones(band_lim)
		np.testing.assert_almost_equal(phi2[(j_max+1)*band_lim:],
			correct_values)

	def test_s2dw_harmonic(self):
		# Test that the harmonic space representation of the wavelet and
		# scaling function kernels are correctly calculated.
		scale_int = 2
		band_lim = 1024
		j_max = wavelets.calc_j_max(band_lim,scale_int)
		phi2 = np.zeros((j_max+2)*band_lim)
		n_quads = 1000
		j_min = 2
		wavelets.phi2_s2dw(phi2,band_lim,scale_int,n_quads)

		wav_har = np.zeros((j_max+2)*band_lim)
		scale_har = np.zeros(band_lim)

		wavelets.s2dw_harmonic(wav_har,scale_har,band_lim,scale_int,n_quads,
			j_min)

		# Test that the scales below j_min are set to zero
		np.testing.assert_equal(wav_har[:j_min*band_lim],
			np.zeros(j_min*band_lim))

		# Test the scales above j_min by hand
		for j in range(j_min,j_max+1):
			true_values = (phi2[(j+1)*band_lim:(j+2)*band_lim]-
				phi2[j*band_lim:(j+1)*band_lim])
			true_values[true_values<0] = 0
			true_values = np.sqrt(true_values)
			ell = np.arange(band_lim)
			true_values *= np.sqrt(2*ell+1)/np.sqrt(4*np.pi)
			np.testing.assert_almost_equal(
				wav_har[j*band_lim:(j+1)*band_lim],true_values)

	def test_get_max_nside(self):
		# Test that the max nside returned is correct
		nside = 128
		scale_int = 3
		j=1
		self.assertEqual(wavelets.get_max_nside(scale_int,j,nside),32)

		# Try a few more values of j and scale_int
		j=2
		self.assertEqual(wavelets.get_max_nside(scale_int,j,nside),32)

		j=3
		self.assertEqual(wavelets.get_max_nside(scale_int,j,nside),32)

		scale_int=2
		j=4
		self.assertEqual(wavelets.get_max_nside(scale_int,j,nside),32)

		# Check that it wont push past nside
		j=9
		self.assertEqual(wavelets.get_max_nside(scale_int,j,nside),128)
		j=10
		self.assertEqual(wavelets.get_max_nside(scale_int,j,nside),128)

	def test_get_alm_indices(self):
		# Test that the alm indices returned are wbat we expect
		old_lmax = 128
		new_lmax = 256

		new_indices = wavelets.get_alm_indices(old_lmax,new_lmax)

		index = 5
		l, m = hp.Alm.getlm(old_lmax,index)
		self.assertEqual(new_indices[index],hp.Alm.getidx(new_lmax,
			l,m))
		index = 100
		l, m = hp.Alm.getlm(old_lmax,index)
		self.assertEqual(new_indices[index],hp.Alm.getidx(new_lmax,
			l,m))
		index = 35
		l, m = hp.Alm.getlm(old_lmax,index)
		self.assertEqual(new_indices[index],hp.Alm.getidx(new_lmax,
			l,m))

	def test_s2dw_wavelet_tranform(self):
		# Check that the python and s2let output match
		input_map_path = self.root_path + 'gmca_test_full_sim_90_GHZ.fits'
		input_map = hp.read_map(input_map_path,dtype=np.float64)
		output_map_prefix = self.root_path + 's2dw_test'
		band_lim = 256
		scale_int = 3
		j_min = 1
		input_fwhm = 1e-10

		# Generate the wavelet maps using the python code
		wavelet_dict = wavelets.s2dw_wavelet_tranform(input_map,
			output_map_prefix,band_lim,scale_int,j_min,input_fwhm,
			n_quads=1000)

		# Check the properties we want are set
		self.assertEqual(wavelet_dict['band_lim'],band_lim)
		self.assertEqual(wavelet_dict['scale_int'],scale_int)
		self.assertEqual(wavelet_dict['j_min'],j_min)
		self.assertEqual(wavelet_dict['j_max'],wavelets.calc_j_max(band_lim,
			scale_int))
		self.assertEqual(wavelet_dict['original_nside'],128)
		self.assertEqual(wavelet_dict['input_fwhm'],input_fwhm)
		self.assertEqual(wavelet_dict['n_scales'],wavelets.calc_j_max(band_lim,
			scale_int)-j_min+2)
		np.testing.assert_equal(wavelet_dict['target_fwhm'],
			np.ones(wavelet_dict['n_scales']+1)*1e-10)

		# Compare the output to the s2let C code output
		scaling_python = hp.read_map(wavelet_dict['scale_map']['path'],
			nest=True)
		scaling_s2let = hp.reorder(hp.read_map(self.root_path +
			'gmca_full_90_GHZ_wav_scal_256_3_1.fits'),r2n=True)
		self.assertLess(np.mean(np.abs(scaling_python-scaling_s2let)),0.001)

		# Repeat the same for the wavelet maps
		for j in range(j_min,wavelet_dict['j_max']+1):
			wav_python = hp.read_map(wavelet_dict['wav_%d_map'%(j)]['path'],
				nest=True)
			wav_s2let = hp.reorder(hp.read_map(self.root_path +
				'gmca_full_90_GHZ_wav_wav_256_3_1_%d.fits'%(j)),r2n=True)
			self.assertLess(np.mean(np.abs(wav_python-wav_s2let)),0.01)

		# Repeat the same test with the precomputed flag
		wavelet_dict = wavelets.s2dw_wavelet_tranform(input_map,
			output_map_prefix,band_lim,scale_int,j_min,input_fwhm,
			n_quads=1000,precomputed=True)

		# Check the properties we want are set
		self.assertEqual(wavelet_dict['band_lim'],band_lim)
		self.assertEqual(wavelet_dict['scale_int'],scale_int)
		self.assertEqual(wavelet_dict['j_min'],j_min)
		self.assertEqual(wavelet_dict['j_max'],wavelets.calc_j_max(band_lim,
			scale_int))
		self.assertEqual(wavelet_dict['original_nside'],128)
		self.assertEqual(wavelet_dict['input_fwhm'],input_fwhm)
		self.assertEqual(wavelet_dict['n_scales'],wavelets.calc_j_max(band_lim,
			scale_int)-j_min+2)
		np.testing.assert_equal(wavelet_dict['target_fwhm'],
			np.ones(wavelet_dict['n_scales']+1)*1e-10)

		# Compare the output to the s2let C code output
		scaling_python = hp.read_map(wavelet_dict['scale_map']['path'],
			nest=True)
		scaling_s2let = hp.reorder(hp.read_map(self.root_path +
			'gmca_full_90_GHZ_wav_scal_256_3_1.fits'),r2n=True)
		self.assertLess(np.mean(np.abs(scaling_python-scaling_s2let)),0.001)

		# Repeat the same for the wavelet maps
		for j in range(j_min,wavelet_dict['j_max']+1):
			wav_python = hp.read_map(wavelet_dict['wav_%d_map'%(j)]['path'],
				nest=True)
			wav_s2let = hp.reorder(hp.read_map(self.root_path +
				'gmca_full_90_GHZ_wav_wav_256_3_1_%d.fits'%(j)),r2n=True)
			self.assertLess(np.mean(np.abs(wav_python-wav_s2let)),0.01)

		# Remove all of the maps we created.
		os.remove(wavelet_dict['scale_map']['path'])
		for j in range(j_min,wavelet_dict['j_max']+1):
			os.remove(wavelet_dict['wav_%d_map'%(j)]['path'])

		# Now we also want to make sure that setting a target beam behaves
		# as we expect
		# Healpy inputs are in radians but arcminutes is a more natural unit
		a2r = np.pi/180/60

		# 1 arcmin input assumed (small)
		input_fwhm = a2r

		# Try outputting at two different resolutions and confirm that
		# things behave as expected
		target_fwhm_big = np.ones(wavelet_dict['n_scales']+1)*a2r*30
		target_fwhm_small = np.ones(wavelet_dict['n_scales']+1)*a2r*20
		output_map_prefix_big = self.root_path + 's2dw_test_big'
		output_map_prefix_small = self.root_path + 's2dw_test_small'

		wavelet_dict_big = wavelets.s2dw_wavelet_tranform(input_map,
			output_map_prefix_big,band_lim,scale_int,j_min,input_fwhm,
			target_fwhm=target_fwhm_big,n_quads=1000)
		np.testing.assert_equal(wavelet_dict_big['target_fwhm'],
			target_fwhm_big)
		wavelet_dict_small = wavelets.s2dw_wavelet_tranform(input_map,
			output_map_prefix_small,band_lim,scale_int,j_min,input_fwhm,
			target_fwhm=target_fwhm_small,n_quads=1000)
		np.testing.assert_equal(wavelet_dict_small['target_fwhm'],
			target_fwhm_small)

		# Set the limit for the comparison to the maximum ell used to
		# write the map.
		scale_lim = min(int(scale_int**j_min),band_lim)
		big_alm = hp.map2alm(hp.reorder(hp.read_map(
			wavelet_dict_big['scale_map']['path'],nest=True),n2r=True),
			lmax=scale_lim)
		big_cl = hp.alm2cl(hp.almxfl(big_alm,1/hp.gauss_beam(
			target_fwhm_big[0],lmax=scale_lim-1)))

		small_alm = hp.map2alm(hp.reorder(hp.read_map(
			wavelet_dict_small['scale_map']['path'],nest=True),n2r=True),
			lmax=scale_lim)
		small_cl = hp.alm2cl(hp.almxfl(small_alm,1/hp.gauss_beam(
			target_fwhm_small[0],lmax=scale_lim-1)))

		# Ignore small cls where numerical error will dominate
		np.testing.assert_almost_equal(big_cl[big_cl>1e-9]/
			small_cl[big_cl>1e-9],np.ones(np.sum(big_cl>1e-9)))

		# Repeat the same comparison for all of the wavelet maps.
		for j in range(j_min,wavelet_dict['j_max']+1):
			wav_lim = min(int(scale_int**(j+1)),band_lim)
			big_alm = hp.map2alm(hp.reorder(hp.read_map(
				wavelet_dict_big['wav_%d_map'%(j)]['path'],nest=True),n2r=True),
				lmax=wav_lim)
			big_cl = hp.alm2cl(hp.almxfl(big_alm,1/hp.gauss_beam(
				target_fwhm_big[0],lmax=wav_lim-1)))

			small_alm = hp.map2alm(hp.reorder(hp.read_map(
				wavelet_dict_small['wav_%d_map'%(j)]['path'],nest=True),n2r=True),
				lmax=wav_lim)
			small_cl = hp.alm2cl(hp.almxfl(small_alm,1/hp.gauss_beam(
				target_fwhm_small[0],lmax=wav_lim-1)))

			# Ignore the last cl, it's zero. Also ignore values that should
			# be 0 since numerical error will dominate.
			np.testing.assert_almost_equal(big_cl[big_cl>1e-9]/
				small_cl[big_cl>1e-9],np.ones(np.sum(big_cl>1e-9)),
				decimal=2)

		# Now make sure that the input beam is also accounted for
		input_fwhm2 = a2r*2
		output_map_prefix_big2 = self.root_path + 's2dw_test_big2'
		wavelet_dict_big2 = wavelets.s2dw_wavelet_tranform(input_map,
			output_map_prefix_big2,band_lim,scale_int,j_min,input_fwhm2,
			target_fwhm=target_fwhm_big,n_quads=1000)

		# Conduct the same comparison with this new factor
		# Set the limit for the comparison to the maximum ell used to
		# write the map.
		scale_lim = min(int(scale_int**j_min),band_lim)
		big_alm = hp.map2alm(hp.reorder(hp.read_map(
			wavelet_dict_big['scale_map']['path'],nest=True),n2r=True),
			lmax=scale_lim)
		big_cl = hp.alm2cl(hp.almxfl(big_alm,hp.gauss_beam(
			input_fwhm,lmax=scale_lim-1)))

		big_alm2 = hp.map2alm(hp.reorder(hp.read_map(
			wavelet_dict_big2['scale_map']['path'],nest=True),n2r=True),
			lmax=scale_lim)
		big_cl2 = hp.alm2cl(hp.almxfl(big_alm2,hp.gauss_beam(
			input_fwhm2,lmax=scale_lim-1)))

		# Ignore small cls where numerical error will dominate
		np.testing.assert_almost_equal(big_cl[big_cl>1e-9]/
			big_cl2[big_cl>1e-9],np.ones(np.sum(big_cl>1e-9)))

		# Repeat the same comparison for all of the wavelet maps.
		for j in range(j_min,wavelet_dict['j_max']+1):
			wav_lim = min(int(scale_int**(j+1)),band_lim)
			big_alm = hp.map2alm(hp.reorder(hp.read_map(
				wavelet_dict_big['wav_%d_map'%(j)]['path'],nest=True),n2r=True),
				lmax=wav_lim)
			big_cl = hp.alm2cl(hp.almxfl(big_alm,hp.gauss_beam(
				input_fwhm,lmax=wav_lim-1)))

			big_alm2 = hp.map2alm(hp.reorder(hp.read_map(
				wavelet_dict_big2['wav_%d_map'%(j)]['path'],nest=True),n2r=True),
				lmax=wav_lim)
			big_cl2 = hp.alm2cl(hp.almxfl(big_alm2,hp.gauss_beam(
				input_fwhm2,lmax=wav_lim-1)))

			# Ignore the last cl, it's zero. Also ignore values that should
			# be 0 since numerical error will dominate.
			np.testing.assert_almost_equal(big_cl[big_cl>1e-9]/
				big_cl2[big_cl>1e-9],np.ones(np.sum(big_cl>1e-9)),
				decimal=2)

		# Remove all of the maps we created.
		os.remove(wavelet_dict_big['scale_map']['path'])
		for j in range(j_min,wavelet_dict_big['j_max']+1):
			os.remove(wavelet_dict_big['wav_%d_map'%(j)]['path'])

		os.remove(wavelet_dict_small['scale_map']['path'])
		for j in range(j_min,wavelet_dict_small['j_max']+1):
			os.remove(wavelet_dict_small['wav_%d_map'%(j)]['path'])

		os.remove(wavelet_dict_big2['scale_map']['path'])
		for j in range(j_min,wavelet_dict_big2['j_max']+1):
			os.remove(wavelet_dict_big2['wav_%d_map'%(j)]['path'])

	def test_s2dw_wavelet_inverse_transform(self):
		# Check that an identity transform using the python wavelet code
		# returns an accurate reconstruction.
		input_map_path = self.root_path + 'gmca_test_full_sim_90_GHZ.fits'
		input_map = hp.read_map(input_map_path,dtype=np.float64)
		output_map_prefix = self.root_path + 's2dw_test'
		band_lim = 300
		scale_int = 3
		j_min = 1
		n_quads = 1000
		input_fwhm = 1e-10
		output_fwhm = 1e-10

		# We want to compare to a map with the same band limit
		input_alm = hp.map2alm(input_map,lmax=band_lim)
		input_map = hp.alm2map(input_alm,nside=128)

		# Generate the wavelet maps using the python code
		wavelet_dict = wavelets.s2dw_wavelet_tranform(input_map,
			output_map_prefix,band_lim,scale_int,j_min,input_fwhm,
			n_quads=n_quads)

		identity_map = wavelets.s2dw_wavelet_inverse_transform(
			wavelet_dict,output_fwhm,n_quads=n_quads)

		self.assertLess(np.mean(np.abs(identity_map-input_map))/
			np.mean(np.abs(input_map)),0.05)

		# Remove all of the maps we created.
		os.remove(wavelet_dict['scale_map']['path'])
		for j in range(j_min,wavelet_dict['j_max']+1):
			os.remove(wavelet_dict['wav_%d_map'%(j)]['path'])

		# Now we'll repeat the same process but with a different
		# set of target beams.
		a2r = np.pi/180/60
		target_fwhm = np.ones(wavelet_dict['n_scales']+1)*a2r*5
		output_fwhm = 1e-10

		wavelet_dict = wavelets.s2dw_wavelet_tranform(input_map,
			output_map_prefix,band_lim,scale_int,j_min,input_fwhm,
			target_fwhm=target_fwhm,n_quads=n_quads)
		identity_map = wavelets.s2dw_wavelet_inverse_transform(
			wavelet_dict,output_fwhm,n_quads=n_quads)

		self.assertLess(np.mean(np.abs(identity_map-input_map))/
			np.mean(np.abs(input_map)),0.05)

		# Remove all of the maps we created.
		os.remove(wavelet_dict['scale_map']['path'])
		for j in range(j_min,wavelet_dict['j_max']+1):
			os.remove(wavelet_dict['wav_%d_map'%(j)]['path'])

		# Now we'll repeat the same process but with a output beam.
		output_fwhm = a2r*30
		wavelet_dict = wavelets.s2dw_wavelet_tranform(input_map,
			output_map_prefix,band_lim,scale_int,j_min,input_fwhm,
			target_fwhm=target_fwhm,n_quads=n_quads)
		fwhm_map = wavelets.s2dw_wavelet_inverse_transform(
			wavelet_dict,output_fwhm,n_quads=n_quads)

		input_fwhm_map = hp.alm2map(hp.almxfl(input_alm,hp.gauss_beam(
			output_fwhm)),nside=128)

		self.assertLess(np.mean(np.abs(fwhm_map-input_fwhm_map))/
			np.mean(np.abs(input_fwhm_map)),0.05)

		# Remove all of the maps we created.
		os.remove(wavelet_dict['scale_map']['path'])
		for j in range(j_min,wavelet_dict['j_max']+1):
			os.remove(wavelet_dict['wav_%d_map'%(j)]['path'])

	def test_multifrequency_wavelet_maps_gmca(self):
		# Test that the multifrequency maps agree with our expectations.
		# We'll do our tests using just two frequencies, and make both
		# frequencie the same map. We'll change the nside to make sure all of
		# the funcitonality is working as intended.
		# Check that the python and s2let output match
		a2r = np.pi/180/60
		input_map_path = self.root_path + 'gmca_test_full_sim_90_GHZ.fits'
		input_map_path_256 = (self.root_path +
			'gmca_test_full_sim_90_GHZ_256.fits')
		input_maps_dict = {
			'30':{'band_lim':256,'fwhm':33*a2r,'path':input_map_path,
				'nside':128},
			'44':{'band_lim':512,'fwhm':5*a2r,'path':input_map_path_256,
				'nside':256}}
		output_maps_prefix = self.root_path + 's2dw_test'
		analysis_type = 'gmca'
		scale_int = 2
		j_min = 1
		wavelets.multifrequency_wavelet_maps(input_maps_dict,
			output_maps_prefix,analysis_type,scale_int,j_min)

		# Generate the wavelet maps using the python code
		wav_analysis_maps = wavelets.multifrequency_wavelet_maps(
			input_maps_dict,output_maps_prefix,analysis_type,scale_int,j_min)

		# Now we can manually acess the maps and make sure everything ended
		# up in the right place. Star with the scale maps.
		# Check the shape for the first group of scales
		n_pix = hp.nside2npix(wavelets.get_max_nside(scale_int,j_min,
			256))
		for j in range(j_min,9):
			n_pix += hp.nside2npix(wavelets.get_max_nside(scale_int,j+1,
				256))
		self.assertTupleEqual(wav_analysis_maps['0'].shape,(2,n_pix))
		# Check the shape for the second group of scales (only includes one
		# scale and one frequency).
		n_pix = hp.nside2npix(wavelets.get_max_nside(scale_int,10,
			256))
		self.assertTupleEqual(wav_analysis_maps['1'].shape,(1,n_pix))

		# Now directly compare the values in the groupped array to the
		# values of the original map. Start with our first
		# frequency
		input_map = hp.read_map(input_map_path,dtype=np.float64,
			verbose=False)
		wavelet_dict = wavelets.s2dw_wavelet_tranform(input_map,
			output_maps_prefix+'t30',input_maps_dict['30']['band_lim'],
			scale_int,j_min,input_maps_dict['30']['fwhm'],
			target_fwhm=np.ones(9)*input_maps_dict['44']['fwhm'])
		n_pix = 0
		nside = wavelets.get_max_nside(scale_int,j_min,512)
		dn = hp.nside2npix(nside)
		scale_map = hp.read_map(wavelet_dict['scale_map']['path'],nest=True)
		np.testing.assert_almost_equal(wav_analysis_maps['0'][0,n_pix:n_pix+dn],
			scale_map)
		for j in range(j_min,9):
			n_pix += dn
			nside = wavelets.get_max_nside(scale_int,j+1,512)
			dn = hp.nside2npix(nside)
			wav_map = hp.ud_grade(hp.read_map(
				wavelet_dict['wav_%d_map'%(j)]['path'],nest=True,
				dtype=np.float64),nside,order_in='NESTED')
			np.testing.assert_almost_equal(
				wav_analysis_maps['0'][0,n_pix:n_pix+dn],wav_map)

		# Now do the same for our second frequency.
		input_map_256 = hp.read_map(input_map_path_256,dtype=np.float64,
			verbose=False)
		wavelet_dict_256 = wavelets.s2dw_wavelet_tranform(input_map_256,
			output_maps_prefix+'t44',input_maps_dict['44']['band_lim'],
			scale_int,j_min,input_maps_dict['44']['fwhm'],
			target_fwhm=np.ones(10)*input_maps_dict['44']['fwhm'])
		n_pix = 0
		nside = wavelets.get_max_nside(scale_int,j_min,512)
		dn = hp.nside2npix(nside)
		scale_map = hp.read_map(wavelet_dict_256['scale_map']['path'],
			nest=True)
		np.testing.assert_almost_equal(wav_analysis_maps['0'][1,n_pix:n_pix+dn],
			scale_map)
		for j in range(j_min,9):
			n_pix += dn
			nside = wavelets.get_max_nside(scale_int,j+1,512)
			dn = hp.nside2npix(nside)
			wav_map = hp.ud_grade(hp.read_map(
				wavelet_dict_256['wav_%d_map'%(j)]['path'],nest=True,
				dtype=np.float64),nside,order_in='NESTED')
			np.testing.assert_almost_equal(
				wav_analysis_maps['0'][1,n_pix:n_pix+dn],wav_map)

		# Delete all the superfluous maps that have been made for testing
		os.remove(wavelet_dict['scale_map']['path'])
		for j in range(j_min,wavelet_dict['j_max']+1):
			os.remove(wavelet_dict['wav_%d_map'%(j)]['path'])

		os.remove(wavelet_dict['scale_map']['path'][:-16]+'30_scaling.fits')
		for j in range(j_min,wavelet_dict['j_max']+1):
			os.remove(wavelet_dict['wav_%d_map'%(j)]['path'][:-14]+
				'30_wav_%d.fits'%(j))

		os.remove(wavelet_dict_256['scale_map']['path'])
		for j in range(j_min,wavelet_dict_256['j_max']+1):
			os.remove(wavelet_dict_256['wav_%d_map'%(j)]['path'])

		os.remove(wavelet_dict_256['scale_map']['path'][:-16]+'44_scaling.fits')
		for j in range(j_min,wavelet_dict_256['j_max']+1):
			os.remove(wavelet_dict_256['wav_%d_map'%(j)]['path'][:-14]+
				'44_wav_%d.fits'%(j))
