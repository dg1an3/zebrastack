open SlopeInterceptObjective

// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.

[<EntryPoint>]
let main argv =     
    optimize |> ignore
    let lossWithTargetIter0 = (loss target iter0)
    LeastSquaresLib.LeastSqOptimizer.optimize lossWithTargetIter0 [initSlope; initOffset]
    0 // return an integer exit code
