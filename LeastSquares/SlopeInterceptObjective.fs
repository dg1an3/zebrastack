module SlopeInterceptObjective

open LeastSquaresLib

// compute current value given current slope intercept parameters
let currentFromSlopeOffset (init:float[]) [slope:float; offset:float] =
    init
    |> Array.map ((*) slope)
    |> Array.map ((+) offset)           
