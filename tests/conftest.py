import os

# CPU keeps the identity suite fast to compile and is what CI runs; a GPU box
# still passes, so this is a default rather than a requirement.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
