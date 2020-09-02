namespace LeastSquaresLib

module AsciiGraph =

    let asciiPixelArray = [|"   "; " . "; " .."; "..."; "..:"; ".::"; ":::"|]

    let asciiGraph (minValue,maxValue) (vector:seq<float>) =
        let index value = 
            float (asciiPixelArray.Length-1)
                * (value - minValue) / (maxValue - minValue + 1.0)
        vector
        |> Seq.map (fun value -> asciiPixelArray.[int (index value)])
        |> String.concat ""

    (* ascii image output *)
    let asciiImage range1d (image:int->int->float) =
        let values = 
            (range1d, range1d) 
            ||> Seq.allPairs
            |> Seq.map (fun (x,y) -> image x y)
        let (minValue, maxValue) = 
            (values |> Seq.min), (values |> Seq.max)
        range1d
        |> Seq.map 
            (fun rowIndex 
                -> asciiGraph (minValue,maxValue) 
                    (range1d |> Seq.map (image rowIndex)))

