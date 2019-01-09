namespace LeastSquaresLib

module ImageVector =

    open System.Diagnostics
    open VectorND
    open LeastSquaresLib
    open LeastSquaresLib.ImageFilters

    type IndexBounds = {xmin:int; ymin:int; xmax:int; ymax:int}

    let imageVectorFromFunc width (imageFunc:ImageFunc) =
        (seq {0..width-1}, seq {0..width-1})
        ||> Seq.allPairs
        |> Seq.map (fun (x,y) -> imageFunc x y)
        |> Array.ofSeq
        |> VectorND

    let imageFuncFromVector width (v:VectorND) x y =
        let index = y*width + x
        if 0 < x && x < width
            && 0 < index && index < v.values.Length
        then v.[index]
        else 0.0

    [<StructuredFormatDisplay("Ascii")>]
    type ImageVector(width:int, vector:VectorND) =        
        do Trace.Assert(vector.values.Length = width * width)
        member this.Width = width
        member this.Vector = vector
        member this.Ascii =
            imageFuncFromVector this.Width this.Vector
            |> AsciiGraph.asciiImage (seq {0..this.Width}) 
            |> String.concat "\n"



    // pyramid from an ImageVector
    type PyramidVector(levels:int, width0:int, vector:VectorND) =
        do Trace.Assert(vector.values.Length = width0 * width0)
        member this.Width0 = width0
        member this.Vector = vector
        member this.levels = levels
        member this.ImageFunc n x y =
            0.0

    let pyramidFromImageVector (iv:ImageVector) =
        iv.Vector
        |> imageFuncFromVector iv.Width
        |> convolve 3 (gauss 2.0)
        |> decimate
        |> imageVectorFromFunc (iv.Width/2) 
        |> function 
            next ->  
            (2, iv.Width, 
                (iv.Vector.values, next.values)
                ||> Array.append 
                |> VectorND)
            |> PyramidVector

    let levelFromPyramid level width0 v =
        width0*width0