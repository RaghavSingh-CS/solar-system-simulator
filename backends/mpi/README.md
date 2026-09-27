# MPI distributed-memory backend

The optional MPI backend uses replicated global body state and deterministic
contiguous target ownership. Each rank computes direct-force accelerations for
its owned targets, then `MPI_Allgatherv` materializes the complete acceleration
vector on every rank before Velocity Verlet advances positions or velocities.
Both force evaluations in every step therefore use globally consistent state.

This baseline retains binary64 direct Newtonian O(N²) physics. Per rank it uses
O(N) memory, approximately O(N²/P) force work, and O(N) communicated
acceleration data per force evaluation. Remainder bodies are assigned one each
to the lowest ranks. When OpenMP is also enabled, local target work may use a
declared thread count; MPI calls remain on the initiating thread.
