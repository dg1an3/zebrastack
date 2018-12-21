namespace LeastSquaresLib

module SlopeInterceptObjective =

    open LeastSquaresLib.LeastSqOptimizer

    // compute current value given current slope/intercept parameters
    let currentFromSlopeOffset (init:SignalType) ([slope:float; offset:float]:OptimizerParameters) =
        init.values
        |> Array.map ((*) slope)
        |> Array.map ((+) offset)
        |> SignalType
