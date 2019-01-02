namespace LeastSquaresLib

module NumericalGradient =

    open Helper
    open VectorND

    type LossFunction = VectorND -> float

    // delta parameter controls numerical approximation of gradient
    let delta = 1e-3

    // gradient of loss function with respect to parameter vector
    let gradient 
            (lossFunc:LossFunction) 
            (atParams:VectorND) : VectorND =

        let loss = lossFunc atParams
        atParams.values
        |> Array.mapi
            (fun outer _
                -> atParams.values
                |> Array.mapi 
                    (fun inner el 
                        -> if (inner = outer) 
                            then el+delta 
                            else el))
        |> Seq.map VectorND
        |> dump "deltas"
        |> Seq.map lossFunc
        |> Seq.map ((+) -loss)
        |> Seq.map ((*) (1.0/delta))
        |> Array.ofSeq
        |> VectorND
