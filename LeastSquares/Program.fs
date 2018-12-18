open LeastSquaresLib
open LeastSquaresLib.LeastSqOptimizer
open LeastSquaresLib.SlopeInterceptObjective

// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.

[<EntryPoint>]
let main argv = 
    let target = [| 0.0; 3.0; 5.0; -2.0; |]
    printfn "target = %A" target

    let iter0 = Helper.genRandomNumbers 4
    printfn "iter0 = %A" iter0

    let initSlope = 1.0
    let initOffset = 0.0

    optimize 
        (quadraticLoss target (currentFromSlopeOffset iter0)) 
            [initSlope; initOffset]
    |> function
        ([finalSlope; finalOffset], finalLoss) ->
            printfn "final: slope = %f; offset = %f; value = %A; loss = %f" 
                finalSlope finalOffset
                (currentFromSlopeOffset iter0 [finalSlope; finalOffset])
                finalLoss

    0 // return an integer exit code
