namespace LeastSquaresLib

module SlopeInterceptObjective =

    open LeastSquaresLib.LeastSqOptimizer

    // compute current value given current slope/intercept parameters
    let currentFromSlopeOffset (init:SignalType) (SlopeIntercept:OptimizerParameters) =
        let [| slope:float; offset:float |] = SlopeIntercept.values
        init.values
        |> Array.map ((*) slope)
        |> Array.map ((+) offset)
        |> SignalType
