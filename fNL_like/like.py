import pyccl as ccl
import numpy as np
import sacc
from scipy.special import spherical_jn
from scipy.interpolate import interp1d
import emcee


def NumberCountsTracerfNL(cosmo, *, z=None, nz=None, bz=None,
                          fNL_term=False, pval=1):
    # First, initialize empty tracer
    tr = ccl.Tracer()
    kernel = ccl.get_density_kernel(cosmo, dndz=(z, nz))
    bz_mean = np.sum(bz*nz)/np.sum(nz)
    sf = (1./(1+z))[::-1]
    if fNL_term:
        k_arr = np.geomspace(1E-4, 1E3, 1536)
        # Calculate transfer function (from EH)
        eh = TkEH(cosmo)
        tk = eh.get_Tk(k_arr)
        # ... or directly from linear power spectrum...
        # pklin = ccl.linear_matter_power(cosmo, k_arr, 1.0)
        # tk = np.sqrt(pklin/k_arr**cosmo['n_s'])
        # tk /= tk[0]

        # Build three different power spectra
        delta_c = 1.686
        Omega_M = cosmo['Omega_m']
        H0 = (cosmo['h']/ccl.physical_constants.CLIGHT_HMPC)
        Dz = ccl.growth_factor(cosmo, sf)
        Dz_norm = 0.01/ccl.growth_factor(cosmo, 0.01)
        Dz = Dz * Dz_norm
        trans_fNL_a = (sf, 3*delta_c*Omega_M*H0**2*(bz[::-1]-pval)/(Dz*(bz_mean-pval)))
        trans_fNL_k = (np.log(k_arr), 1/(k_arr**2*tk))
        tr.add_tracer(cosmo, kernel=kernel, transfer_a=trans_fNL_a,
                      transfer_k=trans_fNL_k)
    else:
        transfer_a = (sf, bz[::-1]/bz_mean)
        tr.add_tracer(cosmo, kernel=kernel, transfer_a=transfer_a)
    return tr


def CMBLensingTracerNonLimber(cosmo, *, z_source, n_samples=3000):
    r"""A Tracer for CMB lensing convergence :math:`\kappa`.
    The associated kernel and transfer function are described
    in Eq. 31 of the `CCL paper <https://arxiv.org/abs/1812.05995>`_.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmology object.
        z_source (:obj:`float`): Redshift of source plane for CMB lensing.
        n_samples (:obj:`int`): number of samples over which the kernel
            is desired. These will be equi-spaced in radial distance.
            The kernel is quite smooth, so usually O(100) samples
            is enough.
    """
    tracer = ccl.Tracer()

    # we need the distance functions at the C layer
    cosmo.compute_distances()
    chi, wchi = ccl.get_kappa_kernel(cosmo, z_source=z_source,
                                     n_samples=n_samples)
    wchi[chi > ccl.comoving_radial_distance(cosmo, 1./(1+6))] = 0
    kernel = (chi, wchi)
    if (cosmo['sigma_0'] == 0):
        tracer.add_tracer(cosmo, kernel=kernel, der_bessel=-1, der_angles=1)
    else:
        tracer._MG_add_tracer(cosmo, kernel, z_source,
                              der_bessel=-1, der_angles=1)
    return tracer


class TkEH(object):
    """ Calculator for the Eisenstein & Hu transfer function.
    """
    def __init__(self, cosmo):
        OMh2 = cosmo['Omega_m']*cosmo['h']**2
        OBh2 = cosmo['Omega_b']*cosmo['h']**2
        self.Om = cosmo['Omega_m']
        self.Ob = cosmo['Omega_b']
        self.h = cosmo['h']
        self.th2p7 = cosmo['T_CMB']/2.7
        # Eq. 2
        self.zeq = 2.5E4*OMh2/self.th2p7**4
        # Eq. 3
        self.keq = 0.0746*OMh2/(cosmo['h']*self.th2p7**2)

        # Eq. 4
        b1 = 0.313*(OMh2**(-0.419))*(1+0.607*OMh2**0.674)
        b2 = 0.238*OMh2**0.223
        self.zdrag=1291*OMh2**0.251*(1+b1*OBh2**b2)/(1+0.659*OMh2**0.828)

        # Eq. 5
        Req = 31.5*OBh2*1000./(self.zeq*self.th2p7**4)
        Rd = 31.5*OBh2*1000/((1+self.zdrag)*self.th2p7**4)
        self.rsound = 2/(3*self.keq)*np.sqrt(6/Req)*np.log((np.sqrt(1+Rd)+np.sqrt(Rd+Req))/(1+np.sqrt(Req)))

        # Eq. 7 (in h/Mpc)
        self.kSilk = 1.6*OBh2**0.52*OMh2**0.73*(1+(10.4*OMh2)**(-0.95))/cosmo['h']

        # Eq. 11
        a1 = (46.9*OMh2)**0.670*(1+(32.1*OMh2)**(-0.532))
        a2 = (12.0*OMh2)**0.424*(1+(45.0*OMh2)**(-0.582))
        self.fb = cosmo['Omega_b']/cosmo['Omega_m']
        self.alphac = (a1**(-self.fb))*(a2**(-self.fb**3))

        # Eq. 12
        bb1 = 0.944/(1+(458*OMh2)**(-0.708))
        bb2 = (0.395*OMh2)**(-0.0266)
        self.betac = 1/(1+bb1*((1-self.fb)**bb2-1))

        y = self.zeq/(1+self.zdrag)
        sqy = np.sqrt(1+y)
        gy = y*(-6*sqy+(2+3*y)*np.log((sqy+1)/(sqy-1)))  # Eq. 15
        self.alphab = 2.07*self.keq*self.rsound*(1+Rd)**(-0.75)*gy  # Eq. 14

        # Eq. 24
        self.betab = 0.5+self.fb+(3-2*self.fb)*np.sqrt((17.2*OMh2)**2+1)

        # Eq. 23
        self.bnode = 8.41*OMh2**0.435

        # Eq. 26
        self.rsound_approx = cosmo['h']*44.5*np.log(9.83/OMh2)/np.sqrt(1+10*OBh2**0.75)

    def _tk_0(self, keq, kh, a, b):
        q = kh/(13.41*keq)
        c = 14.2/a+386./(1+69.9*q**1.08)
        ll = np.log(np.e+1.8*b*q)
        return ll/(ll+c*q**2)

    def _tk_c(self, kh):
        f = 1/(1+(kh*self.rsound/5.4)**4)
        part1 = f*self._tk_0(self.keq, kh, 1, self.betac)
        part2 = (1-f)*self._tk_0(self.keq, kh, self.alphac, self.betac)
        return part1+part2

    def _tk_b(self, kh):
        x = kh*self.rsound
        x_bessel = x/(1+(self.bnode/x)**3)**(1/3)
        part1 = self._tk_0(self.keq, kh, 1, 1)/(1+(x/5.2)**2)
        part2 = self.alphab*np.exp(-(kh/self.kSilk)**1.4)/(1+(self.betab/x)**3)
        return spherical_jn(0, x_bessel)*(part1+part2)

    def _tk_wiggled(self, kh):
        return self.fb*self._tk_b(kh)+(1-self.fb)*self._tk_c(kh)

    def _tk_no_wiggles(self, kh):
        OMh2 = self.Om*self.h**2
        # Eq. 31
        alpha_gamma = 1-0.328*np.log(431*OMh2)*self.fb+0.38*np.log(22.3*OMh2)*self.fb**2
        # Eq. 30
        gamma_eff = self.Om*self.h*(alpha_gamma+(1-alpha_gamma)/(1+(0.43*kh*self.rsound_approx)**4))
        # Eq. 28 (in h/Mpc)
        q = kh*self.th2p7**2/gamma_eff
        # Eq. 29
        l0 = np.log(2*np.e+1.8*q)
        c0 = 14.2+731.0/(1+62.5*q)
        # T_0 in Eq. 29
        return l0/(l0+c0*q**2)

    def get_Tk(self, k, wiggled=True):
        kh = k/self.h
        if wiggled:
            return self._tk_wiggled(kh)
        else:
            return self._tk_no_wiggles(kh)


class LikefNL(object):
    def __init__(self, config,
                 free_params=['b_gal0', 'b_gal1', 'fNL'],
                 fixed_params=['s8']):
        self.nbins = config['nbins']
        self.bins_use = config.get('bins_use', np.arange(self.nbins))
        self.nside = config['nside']
        self.fname_sacc = config['fname_sacc']
        self.fname_extra = config['fname_extra']
        self.prefix_out = config['prefix_out']
        self.lmin_gg = config['lmin_gg']
        self.lmin_gk = config['lmin_gk']
        self.use_gg = config.get('use_gg', True)
        self.use_gk = config.get('use_gk', True)
        self.kmax = config.get('kmax', 0.1)
        self.use_cross = config.get('use_cross', False)
        self.ASN_sigma = config.get('ASN_sigma', 0.1)
        self.priors = {pn: config[f'{pn}_prior'] for pn in free_params}
        self.pfix = {pn: config[f'{pn}_prior'][0] for pn in fixed_params}
        self.par_free_names = free_params
        self.l_limber = config.get('l_limber', -1)
        self.params_all = ['s8', 'fNL'] + [f'b_gal{i}' for i in self.bins_use]
        self.mag_bias = config.get('mag_bias', None)
        self.has_rsd = config.get('has_rsd', False)
        self.bias_normalisation = config.get('bias_normalisation', 1)
        self.bz_model = config.get('bz_model', 'SDSS')
        self.verbose = config.get('verbose', False)
        self.pval = config.get('pval', 1.0)
        self.correct_hartlap = config.get('correct_hartlap', True)
        self.s8_fid = 0.8102

        # Check all params accounted for
        ptot = set(free_params + fixed_params)
        if not ptot == set(self.params_all):
            raise ValueError(f"Declare all parameters {self.params_all} "
                             "as free or fixed")

        # Read ddata
        self.read_sacc()

        # Initialize cosmological stuff
        self.init_cosmo()

    def logprior(self, par):
        for pn, pr in self.priors.items():
            val = par[pn]
            if (val < pr[1]) or (val > pr[2]):
                return -np.inf
        return 0

    def get_params(self, par):
        ptot = par.copy()
        ptot.update(self.pfix)
        return ptot

    def iterate_cls(self):
        for ibin, i in enumerate(self.bins_use):
            for j in self.bins_use[ibin:]:
                if (not self.use_cross) and (j != i):
                    continue
                if self.use_gg:
                    yield f'gal{i}', f'gal{j}'
            if self.use_gk:
                yield f'gal{i}', 'kappa'

    def _get_lmax(self, sacc_tracer):
        cosmo = ccl.CosmologyVanillaLCDM()
        z_mean = np.sum(sacc_tracer.z*sacc_tracer.nz)/np.sum(sacc_tracer.nz)
        chi_mean = ccl.comoving_radial_distance(cosmo, 1/(1+z_mean))
        return self.kmax*chi_mean

    def read_sacc(self):
        # Clean SACC file
        s = sacc.Sacc.load_fits(self.fname_sacc)
        indices = []
        for n1, n2 in self.iterate_cls():
            t1 = s.tracers[n1]
            t2 = s.tracers[n2]
            if (t1.quantity == 'galaxy_density'):
                lmax1 = self._get_lmax(t1)
                if (t2.quantity == 'galaxy_density'):
                    lmax2 = self._get_lmax(t2)
                    lmin, lmax = self.lmin_gg, min(lmax1, lmax2)
                elif (t2.quantity == 'cmb_convergence'):
                    lmin, lmax = self.lmin_gk, lmax1
                else:
                    raise ValueError(f"Quantity {t2.quantity} not supported")
            elif (t1.quantity == 'cmb_convergence'):
                if (t2.quantity == 'galaxy_density'):
                    lmax2 = self._get_lmax(t2)
                    lmin, lmax = self.lmin_gk, lmax2
                else:
                    raise ValueError(
                        f"Quantity pair ({t1.quantity}, {t2.quantity})"
                        "not supported")
            else:
                raise ValueError(f"Quantity {t1.quantity} not supported")
            inds = s.indices(data_type='cl_00', tracers=[n1, n2],
                             ell__lt=lmax, ell__gt=lmin)
            if self.verbose:
                print(n1, n2, lmin, lmax, len(inds))
            indices += inds.tolist()
        s.keep_indices(indices)

        # Get tracer metadata
        self.tracer_data = {}
        for n, t in s.tracers.items():
            q = t.quantity
            if q == 'galaxy_density':
                dndz = (t.z, t.nz)
                lmax = self._get_lmax(t)
            else:
                dndz = None
                lmax = None
            self.tracer_data[n] = {'quantity': q,
                                   'dndz': dndz,
                                   'lmax': lmax}

        # Get Cl metadata
        self.cl_meta = []
        indices = []
        self.data = []
        self.ncls = 0
        for n1, n2 in self.iterate_cls():
            t1 = s.tracers[n1]
            t2 = s.tracers[n2]
            ls, cl, cov, inds = s.get_ell_cl('cl_00', n1, n2,
                                             return_cov=True,
                                             return_ind=True)
            Bbl = s.get_bandpower_windows(inds)
            self.cl_meta.append({'tracer1': n1,
                                 'tracer2': n2,
                                 'leff': ls,
                                 'cl': cl,
                                 'cov': cov,
                                 'inds': inds,
                                 'Bbl_l': Bbl.values,
                                 'Bbl_w': Bbl.weight.T})
            self.data += cl.tolist()
            indices += inds.tolist()
            self.ncls += 1
        self.data = np.array(self.data)
        self.n_data = len(self.data)
        indices = np.array(indices)
        self.cov = s.covariance.dense
        self.cov = self.cov[indices][:, indices]

        if self.correct_hartlap:
            nsims = 1000
            f_hartlap = (nsims-self.n_data-2)/(nsims-1)
        else:
            f_hartlap = 1
        self.icov = f_hartlap * np.linalg.inv(self.cov)

    def init_cosmo(self):
        # Cosmology
        COSMO_P18 = {"Omega_c": 0.26066676,
                     "Omega_b": 0.048974682,
                     "h": 0.6766,
                     "n_s": 0.9665,
                     "sigma8": self.s8_fid}
        self.cosmo = ccl.Cosmology(**COSMO_P18)

        # Deal with magnification bias
        if self.mag_bias is not None:
            zm, sm = np.loadtxt("mag_bias_G20p5.csv", skiprows=1,
                                delimiter=',', unpack=True)
            si = interp1d(zm, sm, bounds_error=False,
                          fill_value=(sm[0], sm[-1]))
        else:
            si = lambda z: 0*z + 0.4

        # Tracers
        for n, d in self.tracer_data.items():
            if d['quantity'] == 'galaxy_density':
                z, nz = d['dndz']
                # Needed for non-Limber stability
                nz[z < 0.01] = 0
                if self.bz_model == 'SDSS':
                    bz = 0.278*((1 + z)**2 - 6.565) + 2.393
                elif self.bz_model == 'Picci':
                    bz = 1.26/ccl.growth_factor(self.cosmo, 1/(1+z))
                else:
                    raise ValueError(f"Unknown bias model {self.bz_model}")
                bz *= self.bias_normalisation
                bz_mean = np.sum(nz*bz)/np.sum(nz)
                self.tracer_data[n]['bz_mean'] = bz_mean

                tr1 = NumberCountsTracerfNL(self.cosmo, z=z, nz=nz, bz=bz,
                                            fNL_term=False)
                tr2 = NumberCountsTracerfNL(self.cosmo, z=z, nz=nz, bz=bz,
                                            fNL_term=True,
                                            pval=self.pval)
                tr3 = ccl.NumberCountsTracer(self.cosmo, has_rsd=self.has_rsd,
                                             dndz=(z, nz), bias=None,
                                             mag_bias=(z, si(z)))
            else:
                # Also needed for non-Limber stability
                tr1 = CMBLensingTracerNonLimber(self.cosmo, z_source=1100,
                                                n_samples=10000)
                tr2 = tr1
                tr3 = tr1
            self.tracer_data[n]['ccltr1'] = tr1
            self.tracer_data[n]['ccltr2'] = tr2
            self.tracer_data[n]['ccltr3'] = tr3

        # Compute templates for each power spectrum
        for clm in self.cl_meta:
            n1 = clm['tracer1']
            n2 = clm['tracer2']
            tr1 = self.tracer_data[n1]
            tr2 = self.tracer_data[n2]
            clm['is_gg'] = ((tr1['quantity'] == 'galaxy_density') and
                            (tr2['quantity'] == 'galaxy_density'))
            t1_1 = tr1['ccltr1']
            t1_2 = tr1['ccltr2']
            t1_3 = tr1['ccltr3']
            t2_1 = tr2['ccltr1']
            t2_2 = tr2['ccltr2']
            t2_3 = tr2['ccltr3']
            ells = clm['Bbl_l']

            cl = ccl.angular_cl(self.cosmo, t1_1, t2_1, ells,
                                l_limber=self.l_limber)
            cl[ells < 2] = 0
            clm['clmm'] = np.dot(clm['Bbl_w'], cl)

            cl = ccl.angular_cl(self.cosmo, t1_1, t2_2, ells,
                                l_limber=self.l_limber)
            cl[ells < 2] = 0
            clm['clmp'] = np.dot(clm['Bbl_w'], cl)

            if self.mag_bias is None:
                cl = np.zeros_like(ells)
            else:
                cl = ccl.angular_cl(self.cosmo, t1_1, t2_3, ells,
                                    l_limber=self.l_limber)
                cl[ells < 2] = 0
            clm['clmk'] = np.dot(clm['Bbl_w'], cl)

            cl = ccl.angular_cl(self.cosmo, t1_2, t2_1, ells,
                                l_limber=self.l_limber)
            cl[ells < 2] = 0
            clm['clpm'] = np.dot(clm['Bbl_w'], cl)

            cl = ccl.angular_cl(self.cosmo, t1_2, t2_2, ells,
                                l_limber=self.l_limber)
            cl[ells < 2] = 0
            clm['clpp'] = np.dot(clm['Bbl_w'], cl)

            if self.mag_bias is None:
                cl = np.zeros_like(ells)
            else:
                cl = ccl.angular_cl(self.cosmo, t1_2, t2_3, ells,
                                    l_limber=self.l_limber)
                cl[ells < 2] = 0
            clm['clpk'] = np.dot(clm['Bbl_w'], cl)

            if self.mag_bias is None:
                cl = np.zeros_like(ells)
            else:
                cl = ccl.angular_cl(self.cosmo, t1_3, t2_1, ells,
                                    l_limber=self.l_limber)
                cl[ells < 2] = 0
            clm['clkm'] = np.dot(clm['Bbl_w'], cl)

            if self.mag_bias is None:
                cl = np.zeros_like(ells)
            else:
                cl = ccl.angular_cl(self.cosmo, t1_3, t2_2, ells,
                                    l_limber=self.l_limber)
                cl[ells < 2] = 0
            clm['clkp'] = np.dot(clm['Bbl_w'], cl)

            if self.mag_bias is None:
                cl = np.zeros_like(ells)
            else:
                cl = ccl.angular_cl(self.cosmo, t1_3, t2_3, ells,
                                    l_limber=self.l_limber)
                cl[ells < 2] = 0
            clm['clkk'] = np.dot(clm['Bbl_w'], cl)

    def get_theory(self, par, return_split=False):
        pdict = self.get_params(par)
        rs82 = (pdict['s8']/self.s8_fid)**2
        fNL = pdict['fNL']
        theory = []
        for clm in self.cl_meta:
            n1 = clm['tracer1']
            n2 = clm['tracer2']
            if clm['is_gg']:
                b1 = pdict[f'b_{n1}']*self.tracer_data[n1]['bz_mean']
                b2 = pdict[f'b_{n2}']*self.tracer_data[n2]['bz_mean']
                tl = (b1*b2*clm['clmm']+  # g-g
                      b1*(b2-self.pval)*fNL*clm['clmp']+  # g-fNL
                      b1*clm['clmk']+  # g-k
                      b2*(b1-self.pval)*fNL*clm['clpm']+  # fNL-g
                      fNL**2*(b1-self.pval)*(b2-self.pval)*clm['clpp']+  # fNL-fNL
                      (b1-self.pval)*fNL*clm['clpk']+  # fNL-k
                      b2*clm['clkm']+  # k-g
                      (b2-self.pval)*fNL*clm['clkp']+  # k-fNL
                      clm['clkk'])*rs82  # kk
            else:
                if n1 == 'kappa':
                    b = pdict[f'b_{n2}']*self.tracer_data[n2]['bz_mean']
                    tl = (b*clm['clmm'] +
                          (b-self.pval)*fNL*clm['clmp']+clm['clmk'])*rs82
                else:
                    b = pdict[f'b_{n1}']*self.tracer_data[n1]['bz_mean']
                    tl = (b*clm['clmm'] +
                          (b - self.pval) * fNL * clm['clpm'] + clm['clkm'])*rs82
            theory.append(tl)
        if return_split:
            return theory
        return np.concatenate(theory)

    def chi2(self, par):
        # Theory prediction
        theory = self.get_theory(par)

        # Compute chi2
        res = self.data-theory
        chi2 = np.dot(res, np.dot(self.icov, res))
        return chi2, theory

    def logprob(self, par):
        chi2, _ = self.chi2(par)
        return -0.5*chi2

    def logp_emcee(self, p):
        par = {n: val for n, val in zip(self.par_free_names, p)}
        logprior = self.logprior(par)
        if logprior == -np.inf:
            return -np.inf
        return self.logprob(par) + logprior

    def run_minimize(self):
        from scipy.optimize import minimize

        # Prepare
        default_values = {f'b_gal{i}': 1.0 for i in range(self.nbins)}
        default_values['fNL'] = 0.0
        default_values['s8'] = self.s8_fid
        p0 = [default_values[n] for n in self.par_free_names]

        res = minimize(lambda p: -self.logp_emcee(p), p0,
                       method='Powell')
        logp = self.logp_emcee(res.x)
        pout = {n: p for n, p in zip(self.par_free_names, res.x)}
        return pout, logp

    def run_emcee(self, nsteps):
        # Prepare
        default_values = {f'b_gal{i}': 1.0 for i in range(self.nbins)}
        default_values['fNL'] = 0.0
        default_values['s8'] = self.s8_fid
        p0 = [default_values[n] for n in self.par_free_names]
        npar = len(p0)
        nwalkers = 2*npar
        pos = (p0 + 0.001 * np.random.randn(nwalkers, npar))

        # Run
        sampler = emcee.EnsembleSampler(nwalkers, npar, self.logp_emcee)
        sampler.run_mcmc(pos, nsteps, progress=True)

        # Save
        chain = sampler.get_chain(flat=True)
        logp = sampler.get_log_prob(flat=True)
        fname_out = f'{self.prefix_out}_chain_emcee.npz'
        np.savez(fname_out, chain=chain, logp=logp, params=self.par_free_names)
        return sampler
