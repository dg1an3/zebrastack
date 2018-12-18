module SlopeInterceptObjective

open LeastSquaresLib

// compute current value given current slope intercept parameters
let currentValueFromSlopeOffset init [slope; offset] =
    init
    |> Array.map ((*) slope)
    |> Array.map ((+) offset)

           
// calculate loss function as squared sum
let loss target init [slope; offset] =
    let update = currentValueFromSlopeOffset init [slope; offset]
    (target, update) 
        ||> Array.map2 (-) 
        |> Array.sumBy (fun x -> x*x)

let target = [| 0.0; 3.0; 5.0; -2.0; |]

let iter0 = Helper.genRandomNumbers 4

printfn "target:        %A" target
printfn "init:          %A" iter0
    
let initSlope = 1.0
let initOffset = 0.0
    
printfn "initial loss %f" (loss target iter0 [initSlope; initOffset])

let update (lossFunc:list<float>->float) ([currentSlope; currentOffset], currentLoss) = 
    let [dSlope; dOffset] = LeastSqOptimizer.dParam_dLoss (loss target iter0) [currentSlope; currentOffset]
    let updateSlope = currentSlope - LeastSqOptimizer.rate * dSlope
    let updateOffset = currentOffset - LeastSqOptimizer.rate * dOffset
    let updateLoss = lossFunc [updateSlope; updateOffset]

    printfn "updated slope = %f (%4.2f %%), offset = %f (%4.2f %%), loss = %f" 
        updateSlope (100.0 * abs(dSlope)/(LeastSqOptimizer.delta + abs(currentSlope)))
        updateOffset (100.0 * abs(dOffset)/(LeastSqOptimizer.delta + abs(currentOffset)))
        updateLoss 

    if abs(updateLoss - currentLoss) < 0.2
    then None
    else Some (([updateSlope; updateOffset], updateLoss), ([updateSlope; updateOffset], updateLoss))

let optimize = 
    Seq.unfold (update (loss target iter0)) ([initSlope; initOffset], loss target iter0 [initSlope; initOffset])
    |> List.ofSeq
    |> List.last
    |> function
        ([finalSlope; finalOffset], finalLoss) ->
            printfn "final values = %A" (currentValueFromSlopeOffset iter0 [finalSlope; finalOffset])
