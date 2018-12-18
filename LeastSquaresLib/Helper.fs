namespace LeastSquaresLib

module Helper =

    let rnd = System.Random()

    let genRandomNumbers count =
        Array.init count (fun _ -> 10.0 * rnd.NextDouble())