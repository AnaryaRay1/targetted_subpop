import os
os.environ["NPROC"]="1"
# os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0" possible chatgpt debugger idk tho lol
os.environ["intra_op_parallelism_threads"]="1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="0"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="false" #changed from false to true 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


import bilby
import gwpopulation
import h5py
import numpy as np
import pandas as pd
from bilby.core.prior import PriorDict
from gwpopulation.experimental.jax import JittedLikelihood, NonCachingModel
from gwpopulation.models.redshift import PowerLawRedshift

from models import three_component_mixture_3_restricted_z as mixture_model # population_model_default as mixture_model
gwpopulation.set_backend("jax")
xp = gwpopulation.utils.xp



posteriors = pd.read_pickle("data_o4b/posteriors.pkl")
import dill


with open("data_o4b/injections.pkl", "rb") as ff:
    injections = dill.load(ff)


model = NonCachingModel(
    model_functions=[
        mixture_model,
        #PowerLawRedshift(cosmo_model="Planck15"),
    ],
)

vt = gwpopulation.vt.ResamplingVT(model=model, data=injections, n_events=len(posteriors)) 

likelihood = gwpopulation.hyperpe.HyperparameterLikelihood(
    posteriors=posteriors,
    hyper_prior=model,
    selection_function=vt,
    maximum_uncertainty=1.0,
)

priors = PriorDict(filename = "priors/mixture_z.prior")
# priors = PriorDict(filename = "priors/default.prior")

parameters = priors.sample()
likelihood.parameters.update(parameters)
print(likelihood.log_likelihood_ratio())
import time
start = time.time()
print(likelihood.log_likelihood_ratio())
print("Elapsed:", time.time() - start, "seconds")
jit_likelihood = JittedLikelihood(likelihood)
jit_likelihood.parameters.update(parameters)
start = time.time()
print(jit_likelihood.log_likelihood_ratio())
print("Elapsed:", time.time() - start, "seconds")
start = time.time()
print(jit_likelihood.log_likelihood_ratio())
print("Elapsed:", time.time() - start, "seconds")


result = bilby.run_sampler(
    likelihood=jit_likelihood,
    use_ratio = True,
    priors=priors,
    sampler="dynesty",
    nlive=500,
    label="gwtc5_mixture_SNq1_chieffminmax3",
    sample="acceptance-walk",
    outdir="results_o4b",
    naccept=6,
    resume=True,
    checkpoint_every = 60,
    save="hdf5")

