open System.Security.Cryptography
open Microsoft.FSharp.Quotations

// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.

[<EntryPoint>]
let main argv =     
    SlopeInterceptObjective.optimize |> ignore

    0 // return an integer exit code
