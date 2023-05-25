---
hide:
  - footer
---

# Welcome to the documentation of JAX GW

This is the documentation of the JAX GW project, which is a collection of
scripts and tools to perform GW analysis in JAX. The initial emphasis is in the
characterization of astrophysical gravitational-wave backgrounds.

!!! note

    The package is still in early development. See below.

## Tasks before initial release

- [ ] Orbits for ground and space interferometers
    - [x] Approximate orbit evolution for ground-based interferometer
    - [x] Approximate orbit evolution for space-borne GW constellations
    - [ ] Specific configuration of currently operational observatories
    - [ ] Initial set of automated tests
- [ ] Response function
    - [x] Timing response of a one-way photon arm (in the flexible adiabatic approximation of Cornish 2004)
    - [x] Response of a round-trip arm
    - [x] Response of an arbitrary photon path
    - [ ] Response of an arbitrary combination of paths
    - [ ] Option to restrict the collection of available pairwise paths
    - [x] Michelson response
    - [ ] TDI responses up to TDI-2.0 for N=3
    - [ ] TDI responses up to TDI-2.0 for arbitrary N
    - [ ] Circularly-permuted path definitions
    - [ ] Initial set of automated tests
- [ ] Overlap Reduction Function (ORF)
    - [ ] Calculate the anisotropic ORF between two channels.
    - [ ] Caclulate the isotropic ORF from its anisotropic counterpart.
    - [ ] Initial set of automated tests
- [ ] Signal Processing
    - [x] Project continuous signal to observatory
    - [ ] Estimate required frequency and temporal resolution (and range) for the pipelines of interest.
    - [ ] Implement noise power spectral densities (PSDs) of current and future observatories.
    - [ ] Generate Point Spread Functions (i.e. Beam Patterns)
    - [ ] Calculate Angular Power Spectra `N_l`
    - [ ] Use PSDs to synthesize noise for arbitrary channels.
    - [ ] Project a Gaussian stochastic signal realization to a channel
    - [ ] Project a Gaussian stochastic signal realization to overlap of channels
    - [ ] Initial set of automated tests
- [ ] Stochastic Gravitational Wave Background (SGWB) Modeling
    - [ ] Store anisotropic SGWB astrophysical kernels in 1st order perturbation theory
    - [ ] Calculate signal `C_ells` from astrophysical kernels and matter power spectra
    - [ ] Calculate shot-noise contributions to the astrophysical signal
    - [ ] Add SGWB isotropic PSDs for sources of interest
    - [ ] Add SGWB anisotropic PSDs for sources of interest
    - [ ] Generate realizations of the SGWB PSDs.
    - [ ] Initial set of automated tests
- [ ] Sky Basis
    - [x] Generate sky basis in a `(theta, phi)` grid.
    - [ ] Generate sky basis in `(ell, m)` space.
    - [ ] Convert between `(ell, m)` and pixel-based grids.
    - [ ] (Maybe) Implement a HEALPix pixelization scheme.
    - [ ] Initial set of automated tests
- [ ] Parameter Estimation
    - [ ] Add Numpyro support
    - [ ] Add isotropic PE pipeline
    - [ ] Add anisotropic PE pipeline in `ell` space.
    - [ ] Initial set of automated tests
- [ ] Waveforms
    - [ ] Add Ripple support
    - [ ] Initial set of automated tests
