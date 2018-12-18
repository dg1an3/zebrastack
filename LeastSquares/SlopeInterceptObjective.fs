module SlopeInterceptObjective

// compute current value given current slope intercept parameters
let currentValueFromSlopeOffset init [slope; offset] =
    init
    |> Array.map ((*) slope)
    |> Array.map ((+) offset)

let target = [| 0.0; 3.0; 5.0; -2.0; |]

let genRandomNumbers count =
    let rnd = System.Random()
    Array.init count (fun _ -> 10.0 * rnd.NextDouble())

let iter0 = genRandomNumbers 4
    
printfn "target:        %A" target
printfn "init:          %A" iter0
    
let initSlope = 1.0
let initOffset = 0.0
       
// calculate loss function as squared sum
let loss [slope; offset] =
    let update = currentValueFromSlopeOffset iter0 [slope; offset]
    (target, update) 
        ||> Array.map2 (-) 
        |> Array.sumBy (fun x -> x*x)
    
printfn "initial loss %f" (loss [initSlope; initOffset])

let dSlope_dLoss [slope; offset] = 
    let d = (loss [slope+LeastSqOptimizer.delta; offset]) - (loss [slope; offset])
    let d = LeastSqOptimizer.delta / if d < 0.0 then d - 0.1 else d + 0.1
    d
    // iter0 |> Array.map ((*) 1.0)

let dOffset_dLoss [slope; offset] =
    let d = (loss [slope; offset+LeastSqOptimizer.delta]) - (loss [slope; offset])
    let d = LeastSqOptimizer.delta / if d < 0.0 then d - 0.1 else d + 0.1
    d
    // iter0 |> Array.map (fun x -> 1.0)
    
let update ([currentSlope; currentOffset], currentLoss) = 
    let dSlope = dSlope_dLoss [currentSlope; currentOffset]
    let dOffset = dOffset_dLoss [currentSlope; currentOffset]
    let updateSlope = currentSlope - LeastSqOptimizer.rate * dSlope
    let updateOffset = currentOffset - LeastSqOptimizer.rate * dOffset
    let updateLoss = loss [updateSlope; updateOffset]

    printfn "updated slope = %f (%4.2f %%), offset = %f (%4.2f %%), loss = %f" 
        updateSlope (100.0 * abs(dSlope)/(LeastSqOptimizer.delta + abs(currentSlope)))
        updateOffset (100.0 * abs(dOffset)/(LeastSqOptimizer.delta + abs(currentOffset)))
        updateLoss 

    if abs(updateLoss - currentLoss) < 0.2
    then None
    else Some (([updateSlope; updateOffset], updateLoss), ([updateSlope; updateOffset], updateLoss))

Seq.unfold update ([initSlope; initOffset], loss [initSlope; initOffset])
|> List.ofSeq
|> List.last
|> function
    ([finalSlope; finalOffset], finalLoss) ->
        printfn "final values = %A" (currentValueFromSlopeOffset iter0 [finalSlope; finalOffset])