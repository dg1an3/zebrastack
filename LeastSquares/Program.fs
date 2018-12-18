open System.Security.Cryptography
open Microsoft.FSharp.Quotations

// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.

[<EntryPoint>]
let main argv = 
    Seq.unfold 
        SlopeInterceptObjective.update 
            ([SlopeInterceptObjective.initSlope; SlopeInterceptObjective.initOffset], 
                SlopeInterceptObjective.loss [SlopeInterceptObjective.initSlope; SlopeInterceptObjective.initOffset])
    |> List.ofSeq
    |> List.last
    |> function
        ([finalSlope; finalOffset], finalLoss) ->
            printfn "final values = %A" (SlopeInterceptObjective.currentValueFromSlopeOffset SlopeInterceptObjective.iter0 [finalSlope; finalOffset])    

    0 // return an integer exit code
