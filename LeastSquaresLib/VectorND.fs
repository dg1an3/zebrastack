namespace LeastSquaresLib (* SparseSomLib *)

module VectorND =

    [<StructuredFormatDisplay("{Ascii}")>]
    type VectorND(values:float[]) =
        member this.values = values
        member this.Item(n) = values.[n]
        member this.Ascii = 
            let (minValue,maxValue) =
                (values |> Seq.min, values |> Seq.max)
            values
            |> AsciiGraph.asciiGraph (minValue, maxValue)
        override this.ToString() = sprintf "%A" values
        static member (+) (l:VectorND,r:VectorND) =
            (l.values,r.values) 
            ||> Array.map2 (+) 
            |> VectorND
        static member (-) (l:VectorND,r:VectorND) = 
            (l.values,r.values) 
            ||> Array.map2 (-) 
            |> VectorND
        static member (*) (l:VectorND,r:float) = 
            l.values 
            |> Array.map ((*) r) 
            |> VectorND
        static member (*) (l:float,r:VectorND) = 
            r.values 
            |> Array.map ((*) l) 
            |> VectorND
        static member (*) (l:VectorND,r:VectorND) = 
            (l.values,r.values) 
            ||> Array.map2 (*) 
            |> Array.sum

    let normL2 (v:VectorND) = v * v
    let normL1 (v:VectorND) = v |> normL2 |> sqrt
    let normL0 (v:VectorND) = 
        v.values 
        |> Array.filter ((<>) 0.0) 
        |> Array.length

    let rnd = System.Random()

    let genRandomVector (min, max) count =
        (fun _ -> min + (max - min) * rnd.NextDouble())
        |> Array.init count 
        |> VectorND