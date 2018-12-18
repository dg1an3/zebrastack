namespace LeastSquaresLib

module LeastSqOptimizer =

    let delta = 0.01
    let rate = 0.95

    let stabilize x =
        if x < 0.0
        then x - delta
        else x + delta

    let dParam_dLoss (lossFunc:list<float>->float) (atParams:list<float>) =
        let loss = lossFunc atParams
        seq { 0..atParams |> List.length }
        |> Seq.map 
            (fun outer 
                -> atParams
                |> List.mapi 
                    (fun inner el 
                        -> if (inner = outer) 
                            then el+delta 
                            else el))
        |> Seq.map lossFunc
        |> Seq.map ((-) loss)
        |> Seq.map stabilize
        |> Seq.map ((/) delta)
        |> List.ofSeq

    let update (lossFunc:list<float>->float) (currentParams:list<float>, currentLoss:float) = 
        let dParams = dParam_dLoss lossFunc currentParams
        let updatedParams = 
            (dParams |> List.map ((*) rate), 
                    currentParams) 
                ||> List.map2 (-)
        let updatedLoss = lossFunc updatedParams

        printfn "updated params = %A (%A %%), loss = %f" 
            updatedParams 
            ((updatedParams, currentParams)
                ||> List.map2 
                    (fun updatedEl currentEl 
                        -> 100.0 * abs(updatedEl)/delta + abs(currentEl)))
            updatedLoss 

        if abs(updatedLoss - currentLoss) < 0.5
        then None
        else Some ((updatedParams, updatedLoss), (updatedParams, updatedLoss))

    let optimize (lossFunc:list<float>->float) initParams = 
        Seq.unfold (update lossFunc) (initParams, lossFunc initParams)
        |> List.ofSeq
        |> List.last
        |> function
            (finalParams, finalLoss) ->
                printfn "final values = %A; final loss = %f" finalParams finalLoss
