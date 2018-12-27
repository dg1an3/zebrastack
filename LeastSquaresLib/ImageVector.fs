namespace LeastSquaresLib

module ImageVector =

    open VectorND

    [<StructuredFormatDisplay("Ascii")>]
    type SignalImage =
        {   width:int; 
            signal:VectorND 
        } with
        member this.ImageFunc x y =
            let index = y*this.width + x
            if 0 < x && x < this.width
                && 0 < index && index < this.signal.values.Length
            then this.signal.[index]
            else 0.0
        member this.Ascii =
            let range1d = seq {0..this.width}
            let values = 
                (range1d, range1d) 
                ||> Seq.allPairs
                |> Seq.map (fun (x,y) -> this.ImageFunc x y)
            let (minValue, maxValue) = 
                (values |> Seq.min), (values |> Seq.max)
            let asciiPixelArray = [|"   "; " . "; " .."; "..."; "..:"; ".::"; ":::"|]
            let index value = 
                float (asciiPixelArray.Length-1)
                    * (value - minValue) / (maxValue - minValue + 1.0)
            let asciiPixel value = 
                asciiPixelArray.[int (index value)]
            range1d
            |> Seq.map (fun row -> 
                    System.String.Join("", 
                        range1d |> Seq.map (fun column -> asciiPixel(this.ImageFunc column row))))
            |> Seq.concat     

    //(* helper to expose a signal as an image *)
    //let imageFromSignal (signalImage:SignalImage) =
    //    signalImage.ImageFunc

    let imageToSignal width imageFunc =
        { width = width;
            signal = 
                (seq {0..width-1}, seq {0..width-1})
                ||> Seq.allPairs
                |> Seq.map (fun (x,y) -> imageFunc x y)
                |> Array.ofSeq
                |> VectorND }        

