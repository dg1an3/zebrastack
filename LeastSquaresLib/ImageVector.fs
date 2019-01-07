namespace LeastSquaresLib

module ImageVector =

    open System.Diagnostics
    open VectorND

    [<StructuredFormatDisplay("Ascii")>]
    type ImageVector(width:int, signal:VectorND) =
        do Trace.Assert(signal.values.Length = width * width)
        member this.width = width
        member this.signal = signal
        member this.ImageFunc x y =
            let index = y*this.width + x
            if 0 < x && x < this.width
                && 0 < index && index < this.signal.values.Length
            then this.signal.[index]
            else 0.0
        member this.Ascii =
            this.ImageFunc
            |> AsciiGraph.asciiImage (seq {0..this.width}) 
            |> String.concat "\n"

    (* helper to expose a signal as an image *)
    let imageFunc (imageVector:ImageVector) =
        imageVector.ImageFunc

    let imageToSignal width imageFunc =
        ImageVector(width, 
                (seq {0..width-1}, seq {0..width-1})
                ||> Seq.allPairs
                |> Seq.map (fun (x,y) -> imageFunc x y)
                |> Array.ofSeq
                |> VectorND)

