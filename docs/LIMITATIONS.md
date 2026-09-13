# Numerical conventions and limits

- Sampling uses positive priority `0.10 + (1-w)*(1-0.5*r)`, denoted gamma in the paper. The separate action score `1-2*abs(action-demand)` is not a sampling weight.
- Text mappings are Identity `s`, Square `s**2`, Square root `sqrt(s)`, and Compression `0.25+0.5*s`. R, M and P receive 0, 0.5 and 1; U remains unresolved. These coordinates describe text composition, not measured behavioral response.
- Continuous cohort degree is calculated per realization before averaging. Replacement equals total minus continuously surviving cohort degree and includes recreated exit ties. Restricted lifetime sums survival at ticks 0–399; time-averaged degree uses trapezoids over ticks 0–400 divided by 400.
- One original mediated baseline history diverged from a later seed-only replay starting at mediation tick 51. Unordered `argpartition` output is a plausible but unconfirmed explanation; the actual cause remains unresolved. Historical observations and the 144 saved confirmation exits are the primary records. New comparisons use coherent reruns in one environment. Exact seed-only replay across environments is not claimed.
- Policy comparisons retain pairing within shared graph seeds. Multiple policies, futures or texts on one graph do not create additional independent graph samples. Joint-forecast superiority is conditional; the overall paired total-degree error interval crosses zero.
- The six controlled messages and their descriptive human ratings do not establish human behavioral benefits or validate an automated language scorer. No external independent reproduction is claimed.

The GitHub preparation retains all model and forecast kernels. Changes to other scripts only create output directories and remove old manuscript asset copying/bibliography generation. Small model copies are retained where earlier experiments or compatibility checks depend on their distinct frozen versions.
