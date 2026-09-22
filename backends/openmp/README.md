# OpenMP backend

Enable with `-DNBODY_ENABLE_OPENMP=ON`. The implementation parallelizes the outer destination-body loop using static scheduling. Each destination acceleration is private to one iteration, while the inner source-body loop retains ascending index order.
