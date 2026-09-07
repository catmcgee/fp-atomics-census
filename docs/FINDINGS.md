# Findings notes

The findings themselves are in the README. This file holds analytic notes
that the README's tables point at.

## Bucket outputs are not a safe tolerance

The batch-shape experiments (`probes/shape/`) show that a request's bits
depend on the shape of every batch it took part in. A verifier that does
not control batching might be tempted to accept any of the K outputs the
request could have produced under the K shape histories a scheduler can
plausibly give it, and to treat a match with any of them as a pass.

That tolerance is not safe against an operator who controls batching.
The operator selects the shape history, so at every step the operator
chooses which of the K admissible outputs the request emits. A request of
T tokens then carries up to T times log2(K) bits of operator-chosen
information in its choice of bucket, invisible to a verifier who accepts
every bucket. That is a covert channel, and its capacity grows with the
number of buckets the verifier tolerates. The remedy is to record the
shape history and verify against that one bucket, or to require a
batch-invariant deployment so that K is 1, not to enumerate buckets.
