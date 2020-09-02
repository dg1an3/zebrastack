namespace LeastSquaresLib

module SlopeInterceptObjective =

    open LeastSquaresLib.VectorND

    // compute current value given current slope/intercept parameters
    let currentFromSlopeOffset (init:VectorND) (SlopeIntercept:VectorND) =
        let [| slope:float; offset:float |] = SlopeIntercept.values
        let ret = init.values
                    |> Array.map (fun initEl -> initEl * slope + offset)
                    |> VectorND
        ret
