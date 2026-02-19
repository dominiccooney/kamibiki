See README.md for a description of the project.

See ~/clients/cline/particlesearch for a project which uses Voyage
embeddings and gitoxide for fast git access. This can be a useful
practical reference for using these services/libraries.

We are going for simplicity and extreme performance. In particular, we
will not store data we can produce quickly by simply querying the git
repository, and we will not copy strings everywhere but instead use
Rust lifetimes effectively to work from shared, read-only references
to the data we need.