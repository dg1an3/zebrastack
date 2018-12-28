namespace LeastSquaresLib

module SlopeInterceptObjective =

    open LeastSquaresLib.VectorND

    // compute current value given current slope/intercept parameters
    let currentFromSlopeOffset (init:VectorND) (SlopeIntercept:VectorND) =
        let [| slope:float; offset:float |] = SlopeIntercept.values
        init.values
        |> Array.map ((*) slope)
        |> Array.map ((+) offset)
        |> VectorND
