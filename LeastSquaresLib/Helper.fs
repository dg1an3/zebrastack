namespace LeastSquaresLib

module Helper =

    let rnd = System.Random()

    let genRandomNumbers (min, max) count =
        Array.init count (fun _ -> min + (max - min) * rnd.NextDouble())