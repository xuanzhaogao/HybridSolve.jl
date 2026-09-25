// HybridSolve C ABI over the vendored HybridMD solver. GPL-3.0.
//
// Exported ABI (version 1), see src/libhybrid.jl:
//   hs_abi_version, hs_solve, hs_result_sizes, hs_copy_results, hs_ssheval
//
// The solver keeps all state in globals (MDpara.h), so calls are not reentrant;
// the Julia side serialises them with HybridSolve.LIB_LOCK.
#include "MDpara.h"
#include "itlin.h"
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

void Coulomb_accelerations_Hybrid(int iprint);
void allocate_arrays();

/* Defined in Coulomb_accelerations_Hybrid.cpp (not declared in MDpara.h):
   the GMRES info block, whose rcode reports convergence (0 = converged). */
extern struct ITLIN_INFO *info;

/* Per-sphere interior permittivities, read by patch 0002. */
double *hs_eps_in = NULL;

/* Copied verbatim from upstream main.cpp: allocate_dynamic() (main.cpp:338-528,
   HybridMD@1f87baf). main.cpp is not vendored because it contains the MD driver. */
static void hs_allocate_dynamic()
{
    int i,j,k;
    M=N_col;
    nphi=2*p;
    ntheta=2*p;
    imnum_thresh=N_col*N*im;
    
    b=new double [M*(p+1)*(2*p+1)*2];
    Bknm1D=new double [M*(p+1)*(2*p+1)*2];
    
    imx = new double[imnum_thresh];
    imy = new double[imnum_thresh];
    imz = new double[imnum_thresh];
    imq = new double[imnum_thresh];
    imind = new int[imnum_thresh];
    
    srcPosarr=new double[3*N];
    srcDenarr=new double[(p+1)*(p+1)*N];
    pot_m=new double[3*N];
    
    sqrtk=new double[p+1];
    
    ionind= new int*[N];
    ionind[0] = new int[N*N];
    for (i=1; i<N; i++)
        ionind[i] = ionind[0] + i*N;
    
    sph_imind= new int*[M];
    sph_imind[0] = new int[imnum_thresh];
    for (i=1; i<M; i++)
        sph_imind[i] = sph_imind[0] + i*imnum_thresh/M;
    
    weights= new double*[2*p];
    weights[0] = new double[4*p*p];
    for (i=1; i<2*p; i++)
        weights[i] = weights[0] + i*2*p;
    
    rnodes=new double**[2*p];
    rnodes[0]=new double*[2*p*2*p];
    for (i=1; i<2*p; i++)
        rnodes[i] = rnodes[0] + i*2*p;
    rnodes[0][0]=new double[4*p*p*3];
    for (i=0; i<2*p; i++)
        for(j=0;j<2*p;j++)
            rnodes[i][j] = rnodes[0][0] + i*2*p*3+j*3;
    
    ynm=new complex**[2*p];
    ynm[0]=new complex*[2*p*(2*p+1)];
    for (i=1; i<2*p; i++)
        ynm[i] = ynm[0] + i*(2*p+1);
    ynm[0][0]=new complex[2*p*(2*p+1)*(p+1)];
    for (i=0; i<2*p; i++)
        for(j=0;j<2*p+1;j++)
            ynm[i][j] = ynm[0][0] + i*(2*p+1)*(p+1)+j*(p+1);
    
    Bknm=new complex**[N_col];
    Bknm[0]=new complex*[N_col*(2*p+1)];
    for (i=1; i<N_col; i++)
        Bknm[i] = Bknm[0] + i*(2*p+1);
    Bknm[0][0]=new complex[N_col*(2*p+1)*(p+1)];
    for (i=0; i<N_col; i++)
        for(j=0;j<2*p+1;j++)
            Bknm[i][j] = Bknm[0][0] + i*(2*p+1)*(p+1)+j*(p+1);
    
    BknmCopy=new complex**[N_col];
    BknmCopy[0]=new complex*[N_col*(2*p+1)];
    for (i=1; i<N_col; i++)
        BknmCopy[i] = BknmCopy[0] + i*(2*p+1);
    BknmCopy[0][0]=new complex[N_col*(2*p+1)*(p+1)];
    for (i=0; i<N_col; i++)
        for(j=0;j<2*p+1;j++)
            BknmCopy[i][j] = BknmCopy[0][0] + i*(2*p+1)*(p+1)+j*(p+1);
    
    Bcs=new double**[N_col];
    Bcs[0]=new double*[N_col*(2*p+1)];
    for (i=1; i<N_col; i++)
        Bcs[i] = Bcs[0] + i*(2*p+1);
    Bcs[0][0]=new double[N_col*(2*p+1)*(p+1)];
    for (i=0; i<N_col; i++)
        for(j=0;j<2*p+1;j++)
            Bcs[i][j] = Bcs[0][0] + i*(2*p+1)*(p+1)+j*(p+1);
    
    target= new double*[M*4*p*p];
    target[0] = new double[M*12*p*p];
    for (i=1; i<M*4*p*p; i++)
        target[i] = target[0] + i*3;
    
    fldtarg= new complex*[M*4*p*p];
    fldtarg[0] = new complex[M*12*p*p];
    for (i=1; i<M*4*p*p; i++)
        fldtarg[i] = fldtarg[0] + i*3;
    
    pottarg=new complex[M*4*p*p];
    
    fgrid=new complex**[N_col];
    fgrid[0]=new complex*[N_col*2*p];
    for (i=1; i<N_col; i++)
        fgrid[i] = fgrid[0] + i*2*p;
    fgrid[0][0]=new complex[N_col*4*p*p];
    for (i=0; i<N_col; i++)
        for(j=0;j<2*p;j++)
            fgrid[i][j] = fgrid[0][0] + i*(2*p)*(2*p)+j*(p*2);
    
    ycoef=new complex**[N_col];
    ycoef[0]=new complex*[N_col*(2*p+1)];
    for (i=1; i<N_col; i++)
        ycoef[i] = ycoef[0] + i*(2*p+1);
    ycoef[0][0]=new complex[N_col*(2*p+1)*(p+1)];
    for (i=0; i<N_col; i++)
        for(j=0;j<2*p+1;j++)
            ycoef[i][j] = ycoef[0][0] + i*(2*p+1)*(p+1)+j*(p+1);
    
    Mnodes=new double***[N_col];
    Mnodes[0]=new double**[N_col*2*p];
    for (i=1; i<N_col; i++)
        Mnodes[i] = Mnodes[0] + i*2*p;
    Mnodes[0][0]=new double*[N_col*4*p*p];
    for (i=0; i<N_col; i++)
        for(j=0;j<2*p;j++)
            Mnodes[i][j] = Mnodes[0][0] + i*(2*p)*(2*p)+j*(p*2);
    Mnodes[0][0][0]=new double[N_col*4*p*p*3];
    for (i=0; i<N_col; i++)
        for(j=0;j<2*p;j++)
            for(k=0;k<2*p;k++)
                Mnodes[i][j][k] = Mnodes[0][0][0]+i*(2*p)*(2*p)*3+j*(p*2)*3+k*3;
    
    mpole=new complex***[N_col];
    mpole[0]=new complex**[N_col*N_col];
    for (i=1; i<N_col; i++)
        mpole[i] = mpole[0] + i*N_col;
    mpole[0][0]=new complex*[N_col*N_col*(2*p+1)];
    for (i=0; i<N_col; i++)
        for(j=0;j<N_col;j++)
            mpole[i][j] = mpole[0][0] + i*N_col*(2*p+1)+j*(p*2+1);
    mpole[0][0][0]=new complex[N_col*N_col*(2*p+1)*(p+1)];
    for (i=0; i<N_col; i++)
        for(j=0;j<N_col;j++)
            for(k=0;k<2*p+1;k++)
                mpole[i][j][k]=\
		mpole[0][0][0]+i*M*(2*p+1)*(p+1)+j*(p*2+1)*(p+1)+k*(p+1);
    
    local=new complex***[N_col];
    local[0]=new complex**[N_col*N_col];
    for (i=1; i<N_col; i++)
        local[i] = local[0] + i*N_col;
    local[0][0]=new complex*[N_col*N_col*(2*p+1)];
    for (i=0; i<N_col; i++)
        for(j=0;j<N_col;j++)
            local[i][j] = local[0][0] + i*N_col*(2*p+1)+j*(p*2+1);
    local[0][0][0]=new complex[N_col*N_col*(2*p+1)*(p+1)];
    for (i=0; i<N_col; i++)
        for(j=0;j<N_col;j++)
            for(k=0;k<2*p+1;k++)
                local[i][j][k] =\
		local[0][0][0] + i*M*(2*p+1)*(p+1)+j*(p*2+1)*(p+1)+k*(p+1);
    
    source= new double*[imnum_thresh+N];
    source[0] = new double[(imnum_thresh+N)*3];
    for (i=1; i<imnum_thresh+N; i++)
        source[i] = source[0] + i*3;
    
    mpsource= new double*[N-N_ion];
    mpsource[0] = new double[(N-N_ion)*3];
    for (i=1; i<N-N_ion; i++)
        mpsource[i] = mpsource[0] + i*3;
    
    dipvec= new double*[imnum_thresh+N];
    dipvec[0] = new double[(imnum_thresh+N)*3];
    for (i=1; i<imnum_thresh+N; i++)
        dipvec[i] = dipvec[0] + i*3;
    
    mpdipvec= new double*[N_col];
    mpdipvec[0] = new double[(N_col)*3];
    for (i=1; i<N_col; i++)
        mpdipvec[i] = mpdipvec[0] + i*3;
    
    
    fld= new complex*[imnum_thresh+N];
    fld[0] = new complex[(imnum_thresh+N)*3];
    for (i=1; i<imnum_thresh+N; i++)
        fld[i] = fld[0] + i*3;
    
    charge=new complex[imnum_thresh+N];
    dipstr=new complex[imnum_thresh+N];
    mpdipstr=new complex[N_col];
    mptarget=new double[3];
    pot=new complex[imnum_thresh+N];
    scarray=new double[10*(p+2)*(p+2)];
    wlege=new double[10*(p+2)*(p+2)];
}

extern "C" int hs_abi_version(void) { return 1; }

extern "C" int hs_solve(int ns, const double *centers, const double *radii, const double *eps_r,
                        int nq, const double *qpos, const double *qv,
                        int p_, int im_, double gmres_tol, int fmm_iprec,
                        double source_tol, double sph_tol, int verbose)
{
    if (ns < 1 || nq < 1 || p_ < 1 || im_ < 2) return 3;
    if (!centers || !radii || !eps_r || !qpos || !qv) return 3;
    N = nq + ns; Ntype = 2;
    N_ion = nq; N_ion1 = nq; N_ion2 = 0;
    N_col = ns; M = ns; N_col1 = ns; N_col2 = 0;
    allocate_arrays();
    for (int i = 0; i < nq; i++) {
        x[i] = qpos[3*i]; y[i] = qpos[3*i+1]; z[i] = qpos[3*i+2];
        q[i] = qv[i]; r[i] = 0.0;
    }
    for (int s = 0; s < ns; s++) {
        int k = nq + s;
        x[k] = centers[3*s]; y[k] = centers[3*s+1]; z[k] = centers[3*s+2];
        q[k] = 0.0; r[k] = radii[s];
    }
    delete[] hs_eps_in; hs_eps_in = new double[ns];
    for (int s = 0; s < ns; s++) hs_eps_in[s] = eps_r[s];
    c_lj = 0.0; epsi_ion = 1.0;
    p = p_; im = im_; imm = im_;
    gmrestol = gmres_tol; fmmtol = fmm_iprec;
    sourcetol = source_tol; sphtol = sph_tol;
    iter_indicator = 0; force_compute = 0; energy_compute = 1;
    hs_allocate_dynamic();

    int saved = -1, devnull = -1;
    fflush(stdout);
    if (!verbose) {
        saved = dup(1); devnull = open("/dev/null", O_WRONLY);
        if (saved >= 0 && devnull >= 0) dup2(devnull, 1);
    }
    Coulomb_accelerations_Hybrid(0);
    fflush(stdout);
    if (!verbose) {
        if (saved >= 0 && devnull >= 0) dup2(saved, 1);
        if (saved >= 0) close(saved);
        if (devnull >= 0) close(devnull);
    }
    /* gmres() sets info->rcode: 0 converged, 2 maxiter exceeded, other nonzero
       values are allocation/argument/QR failures (GMRES/gmres.c header). */
    if (info == NULL || info->rcode != 0) return 2;
    return 0;
}

static int hs_count_images()
{
    /* imcount is local to Coulomb_accelerations_Hybrid; every (source i, sphere j)
       pair with ionind[i][j] == 1 contributed exactly im images. */
    int pairs = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            if (ionind[i][j] == 1) pairs++;
    return pairs * im;
}

extern "C" int hs_result_sizes(int *nimages, int *p_out, int *ns_out)
{
    *nimages = hs_count_images(); *p_out = p; *ns_out = M;
    return 0;
}

extern "C" int hs_copy_results(double *ox_, double *oy_, double *oz_, double *oq_, int *osph,
                               double *bknm, double *energy)
{
    int n = hs_count_images();
    for (int i = 0; i < n; i++) {
        ox_[i] = imx[i]; oy_[i] = imy[i]; oz_[i] = imz[i]; oq_[i] = imq[i];
        osph[i] = imind[i] - N_ion;
    }
    memcpy(bknm, &Bknm[0][0][0], sizeof(complex) * M * (2*p+1) * (p+1));
    *energy = printed_ele_energy;
    return 0;
}

extern "C" void ssheval_(complex *, int *, double [3], complex *);

extern "C" void hs_ssheval(const double *ycoef, int p_, const double *target3, double *out)
{
    double t[3] = {target3[0], target3[1], target3[2]};
    complex f;
    ssheval_((complex *)ycoef, &p_, t, &f);
    out[0] = f.real; out[1] = f.imag;
}
