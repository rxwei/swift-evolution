# Differentiable programming for gradient-based machine learning

* Proposal: [SE-NNNN](NNNN-filename.md)
* Authors: [Richard Wei], [Dan Zheng], [Marc Rasi], [Bart Chrzaszcz], [Aleksandr Efremov]
* Review Manager: TBD
* Status: **Pitch**
* Implementation: On `main` branch behind `import _Differentiation`

*During the review process, add the following fields as needed:*

* Decision Notes: [Rationale](https://forums.swift.org/), [Additional Commentary](https://forums.swift.org/)
* Bugs: [SR-NNNN](https://bugs.swift.org/browse/SR-NNNN), [SR-MMMM](https://bugs.swift.org/browse/SR-MMMM)
* Previous Revision: [1](https://github.com/apple/swift-evolution/blob/...commit-ID.../proposals/NNNN-filename.md)
* Previous Proposal: [SE-XXXX](XXXX-filename.md)

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Differentiable programming for gradient-based machine learning](#differentiable-programming-for-gradient-based-machine-learning)
    - [Introduction](#introduction)
        - [Example: Intelligent apps](#example-intelligent-apps)
    - [Motivation](#motivation)
        - [Type-safe machine learning](#type-safe-machine-learning)
        - [Calculus is fun](#calculus-is-fun)
            - [Animations](#animations)
            - [Games](#games)
            - [Simulations](#simulations)
            - [Robotics](#robotics)
            - [Rendering and ray tracing](#rendering-and-ray-tracing)
    - [Math introduction](#math-introduction)
        - [What is a derivative?](#what-is-a-derivative)
        - [Iterative optimization](#iterative-optimization)
        - [Derivatives of functions with arbitrary inputs](#derivatives-of-functions-with-arbitrary-inputs)
    - [History of differentiation algorithms](#history-of-differentiation-algorithms)
        - [Numerical differentiation](#numerical-differentiation)
        - [Symbolic differentiation](#symbolic-differentiation)
        - [Automatic differentiation](#automatic-differentiation)
    - [Approaches to automatic differentiation](#approaches-to-automatic-differentiation)
        - [Embedded domain-specific languages](#embedded-domain-specific-languages)
        - [Source code transformation tools](#source-code-transformation-tools)
        - [First-class language support](#first-class-language-support)
        - [Why bake differentiation into Swift?](#why-bake-differentiation-into-swift)
            - [Maximal coverage of Swift language features](#maximal-coverage-of-swift-language-features)
            - [Extensibility](#extensibility)
            - [Static warnings and errors](#static-warnings-and-errors)
            - [The pursuit for user-defined code transformations](#the-pursuit-for-user-defined-code-transformations)
    - [Proposed solution](#proposed-solution)
        - [The `Differentiable` protocol](#the-differentiable-protocol)
        - [The `@differentiable(reverse)` declaration attribute](#the-differentiablereverse-declaration-attribute)
        - [`@differentiable(reverse)` function types](#differentiablereverse-function-types)
        - [`@derivative` attribute](#derivative-attribute)
        - [Differential operators](#differential-operators)
    - [Detailed design](#detailed-design)
        - [Differentiable data structures](#differentiable-data-structures)
            - [The `Differentiable` protocol](#the-differentiable-protocol-1)
            - [`Differentiable` conformances](#differentiable-conformances)
            - [Compiler-synthesized conformances](#compiler-synthesized-conformances)
                - [Synthesis conditions](#synthesis-conditions)
                - [Default synthesis](#default-synthesis)
                    - [Opt out of synthesis for a stored property](#opt-out-of-synthesis-for-a-stored-property)
                - [Shortcut synthesis](#shortcut-synthesis)
        - [Differentiable function declarations](#differentiable-function-declarations)
            - [The `@differentiable(reverse)` declaration attribute](#the-differentiablereverse-declaration-attribute-1)
            - [Conformance and subclassing](#conformance-and-subclassing)
                - [Protocol dispatch](#protocol-dispatch)
                - [Class dispatch](#class-dispatch)
        - [Make a function differentiable using `@derivative`](#make-a-function-differentiable-using-derivative)
            - [Derivative functions](#derivative-functions)
                - [Typing rules](#typing-rules)
                    - [Differentiability parameters](#differentiability-parameters)
                    - [Differentiability generic requirements](#differentiability-generic-requirements)
            - [Access control](#access-control)
        - [Differentiable function types](#differentiable-function-types)
            - [Function subtyping and runtime representation](#function-subtyping-and-runtime-representation)
            - [The `@differentiable(reverse)` function type attribute](#the-differentiablereverse-function-type-attribute)
            - [Type conversion](#type-conversion)
                - [Coercing function declarations into `@differentiable(reverse)` function values](#coercing-function-declarations-into-differentiablereverse-function-values)
                - [Upcasting to non-`@differentiable(reverse)` functions](#upcasting-to-non-differentiablereverse-functions)
            - [Non-differentiable parameters](#non-differentiable-parameters)
        - [Differential operators](#differential-operators-1)
            - [`gradient(of:)`](#gradientof)
            - [`gradient(at:in:)`](#gradientatin)
            - [`valueWithGradient(at:in:)`](#valuewithgradientatin)
            - [`valueWithPullback(at:in:)`](#valuewithpullbackatin)
        - [Static analysis](#static-analysis)
            - [Cross-module opacity](#cross-module-opacity)
            - [Non-differentiable type conversions](#non-differentiable-type-conversions)
            - [Accidental data flow mistakes](#accidental-data-flow-mistakes)
    - [Source compatibility](#source-compatibility)
    - [Effect on ABI stability](#effect-on-abi-stability)
    - [Effect on API resilience](#effect-on-api-resilience)
        - [`Differentiable` protocol](#differentiable-protocol)
        - [Differential operators](#differential-operators-2)
    - [Alternatives considered](#alternatives-considered)
        - [Not support differentiable programming](#not-support-differentiable-programming)
        - [Use another language or framework for differentiable programming](#use-another-language-or-framework-for-differentiable-programming)
        - [Other approaches to differentiable programming](#other-approaches-to-differentiable-programming)
    - [Acknowledgements](#acknowledgements)

<!-- markdown-toc end -->


## Introduction

Derivatives are a fundamental tool in calculus and have applications in many
domains, notably gradient-based machine learning (ML). As an easy-to-use,
high-performance language, Swift is a great fit for both highly expressive
algorithms and numerical computations. Meanwhile, ML is one of the fastest
growing technologies in modern days, but the mainstream ML development tools are
mostly based on dynamic languages where it can be challenging for developers to
take advantange of software debugging tools and compile-time code diagnostics or
to maintain type safety in large-scale software.

As a compiled programming language with a modern type system, Swift has a unique
opportunity to develop its own numerical computing and ML ecosystem. Driven by
the growing needs of ML libraries and algorithms, we believe one key technology,
differentiable programming, will help push ML development experience and
developer productivity to a whole new level.

We propose adding differentiable programming as a first-class,
language-integrated feature in Swift, making Swift become the first
general-purpose, statically-typed programming language to have [automatic
differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
capabilities.

At a glance, this feature includes the following additions:

- A `@differentiable(reverse)` declaration attribute for declaring
  differentiable functions.
- `@differentiable(reverse)` function types.
- A `@derivative(of:)` attribute for defining custom derivatives.
- A `Differentiation` module to be distributed in Swift releases, containing:
  - A `Differentiable` protocol, generalizing data structures that are
    differentiable.
  - Differential operators (e.g. `gradient(of:)`), for evaluating the
    derivatives of functions.

Differentiable programming is a new paradigm for programming in which programs
can be differentiated throughout. At a glance, differentiable programming lets
you take the derivative of functions whose parameters and results conform to the
`Differentiable` protocol.

```swift
import Differentiation

func f(_ x: SIMD32<Float>) -> Float {
    (x * x).sum()
}
let dfdx = gradient(of: f)
dfdx(SIMD32(repeating: 3)) // SIMD32([6, 6, 6, 6, ...])
```

The ability to get derivatives of programs enables a new world of numerical
computing applications, notably machine learning. With first-class support,
gradient-based learning algorithms can even be built using standard library
types such as `Float` and `SIMD64<Float>` and be differentiated using
protocol-oriented APIs such as `valueWithGradient(at:in:)`.

```swift 
import Differentiation

struct Perceptron: Differentiable {
    var weight: SIMD2<Float> = .random(in: -1..<1)
    var bias: Float = 0

    func callAsFunction(_ input: SIMD2<Float>) -> Float {
        (weight * input).sum() + bias
    }
}

var model = Perceptron()
let andGateData: [(x: SIMD2<Float>, y: Float)] = [
    (x: [0, 0], y: 0),
    (x: [0, 1], y: 0),
    (x: [1, 0], y: 0),
    (x: [1, 1], y: 1),
]
for _ in 0..<100 {
    let (loss, modelGradient) = valueWithGradient(at: model) { model -> Float in
        var loss: Float = 0
        for (x, y) in andGateData {
            let prediction = model(x)
            let error = y - prediction
            loss = loss + error * error / 2
        }
        return loss
    }
    print(loss)
    model.weight -= modelGradient.weight * 0.02
    model.bias -= modelGradient.bias * 0.02
}
```

Differentiable programming scales up from simple examples like this to
full-fledged machine learning models using neural networks. Neural networks are
similar to the `Perceptron` example above in that it contains trainable
parameters (commonly part of neural network layers) and each parameter can be
modified based on gradient of a loss with respect to each parameter. Neural
network layers can be generalized by a protocol that inherits from
`Differentiable`:

```swift
// Example library:
public protocol Layer: Differentiable {
    associatedtype Input: Differentiable
    associatedtype Output: Differentiable

    @differentiable(reverse)
    func callAsFunction(_ input: Input) -> Output
}

public class Dense: Layer { ... }
public class Convolution: Layer { ... }
public struct NDArray: Differentiable { ... }

// Client code:
final class MyModel: Layer {
    let dense1: Dense
    let dense2: Dense

    func callAsFunction(_ input: NDArray<Float>) -> NDArray<Float> {
        dense2(dense1(input))
    }
}
```

While the differentiation APIs are flexible and fully dynamic, differentiation
is based on a program transformation that happens at compile time. This enables
many static analyses that not only help produce more efficient code but also
detect common numerical programming mistakes such as non-differentiable
functions and zero derivatives.

```console
let grad = gradient(at: 1.0) { x in
    3.0.squareRoot()
}
```

```console
test.swift:2:4: warning: result does not depend on differentiation arguments and will always have a zero derivative
    3.0.squareRoot()
    ^
test.swift:2:4: note: add 'withoutDerivative(at:)' to silence the warning if zero derivatives are intentional
    3.0.squareRoot()
    ^
    withoutDerivative(at:  )
```

Unlike library-based automatic differentiation, differentiable programming makes
many common runtime errors in machine learning become directly debuggable using
LLDB without library boundaries. Also contrary to library-based approaches,
differential operators offered in the `Differentiation` library can be used to
take the derivative of functions on any type that conforms to the
`Differentiable` protocol, such as `Float`, `SIMD4<Double>`, `Complex<Double>`,
`[Float]` and custom types. This enables programmers to integrate gradient-based
learning algorithms, physical simulations, and scientific experiments directly
in their applications without having to incorporate any embedded domain-specific
language or an automatic differentiation algorithm.

### Example: Intelligent apps

One example that uses gradient-based machine learning techniques to enhance user
experiences of an app is providing intellience based on learned user behavior.
Intelligent apps can make predictions, provide suggestions, and learn user
preferences: all of these can be powered by differentiable programming.

The core of such an intelligent app is a function with real-valued "trainable
parameters". Differentiation can be used to systematically optimize (i.e. find
"good" values for) these parameters via gradient descent. (Optimizing these
parameters via conventional algorithms is typically difficult or intractable.)

Consider a podcast player that tries to automatically adjust the playback speed
based on the podcast type and the podcast section. We can define its business
logic as the following, as well as a "model" which contains real-valued
parameters that control how inputs get mapped onto outputs.

```swift
enum PodcastCategory: Int {
    case comedy
    case news
    ...
}

enum PodcastSection: Int {
    case advertisement
    case introduction
    case body
    case conclusion
}

struct PodcastState {
    let category: PodcastCategory
    let section: PodcastSection
}

struct PodcastSpeedModel: Differentiable {
    var minSpeed, maxSpeed: Float
    /// The multiplier for each podcast category.
    var categoryMultipliers: [Float] 
    /// The multiplier for each podcast section.
    var sectionMultipliers: [Float]

    /// Returns a podcast speed multiplier prediction for the given podcast category
    /// and section.
    func prediction(for state: PodcastState) -> Float {
        let speed = categoryMultipliers[state.category] * sectionMultipliers[state.section]
        if speed < minSpeed { return minSpeed }
        if speed > maxSpeed { return maxSpeed }
        return speed
    }
}
```

Parameters in this podcast speed model, represented as stored properties in the
struct, determine how quickly the podcast should play under different
circumstances: `minSpeed`, `maxSpeed`, `categoryMultipliers`, and
`sectionMultipliers`. A priori, it is not clear what good parameter values are,
and different users may prefer different parameter values.

An intelligent application could determine personalized parameter values as
follows:

1.  Let the user set the speed manually, and record observations whenever the
    user changes the speed.

2.  After collecting enough observations, search for parameter values such that
    the model predicts speeds close to the user's preferred speed. If such
    values are found, offer to start automatically setting the speed.

"Gradient descent" is an algorithm that performs this search, and a language
that supports differentiable programming makes it easy to implement gradient
descent. Here is some pseudocode illustrating gradient descent.

First, we need an objective function for gradient descent to minimize.
[Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error) is used
here:

```swift
struct Observation {
    var podcastState: PodcastState
    var userSpeed: Float
}

func meanError(for model: PodcastSpeedModel, _ observations: [Observation]) -> Float {
    var error: Float = 0
    for observation in observations {
        error += abs(model.prediction(for: observation.podcastState) - observation.userSpeed)
    }
    return error / Float(observations.count)
}
```

Next, we implement the gradient descent algorithm. In the loop, we take the
gradient of the mean error with respect to the model (i.e. with respect to its
properties such as `minSpeed` and `categoryMultipliers`). After some iterations,
the mean error will be minimized and the model will produce more "correct"
results based on its learning.

```swift
var model = PodcastSpeedModel()
let observations = storage.observations()
for _ in 0..<1000 {
    // The language differentiates `meanError` to get a "gradient", which is a value indicating
    // how to change `model` in order to decrease the value of `meanError`.
    let modelGradient = gradient(at: model) { meanError(for: $0, observations) }

    // Change `model` in the direction that decreased the value of `meanError`.
    let learningRate = 0.01
    model.minSpeed -= learningRate * modelGradient.minSpeed
    model.maxSpeed -= learningRate * modelGradient.maxSpeed
    for i in model.categoryMultipliers.indices {
        model.categoryMultipliers[i] -= learningRate * modelGradient.categoryMultipliers[i]
    }
    for i in model.sectionMultipliers.indices {
        model.sectionMultipliers[i] -= learningRate * modelGradient.sectionMultipliers[i]
    }
}
```

As we can see, differentiable programming enables developers to effortlessly
incorporate extremely lightweight gradient-based learning algorithms into
applications, while having derivative code synthesized automatically by Swift.

Language-integrated differentiable programming benefits not only ML
practitioners and app developers, but also developers of ML and scientific
computing frameworks. Relying on a single language-integrated differentiable
programming eliminates the burden of separately maintaining an automatic
differentiation algorithm and a domain-specific langauge, easing the development
and maintenance overhead.

## Motivation

We believe that first-class differentiable programming is a big step towards
high-level numerical computing support and will make Swift a real contender in
the numerical computing and machine learning landscape. Differentiable
programming will enable intelligent applications, machine learning models,
scientific experiments, physical simulations, and more.

### Type-safe machine learning

Today, machine learning is predominantly done in dynamically-typed languages
like Python: these languages are concise and easy to use. However, some people
prefer safer programming: features like type checking and static diagnostics
help catch errors early and improve productivity.

Differentiable programming in Swift enables safe, expressive machine learning.
Custom differentiable data structures can be declared and checked at
compile time. Thanks to protocol-oriented programming, differentiable types are
generalized by a protocol, enabling differential operators to be defined as
higher-order functions constrained on such a protocol. Mathematical optimization
algorithms such as neural network optimizers can also be defined generically
over such a protocol and work with all differentiable types.

### Calculus is fun

Calculus is fun, and differentiation in the Swift toolbox will let programmers
explore that fun. Here are some interesting applications:

#### Animations

[Easing functions](https://stackoverflow.com/a/8317722) specify the rate of
change of parameters for animations. Differentiation enables easy manipulation
of these functions.

#### Games

Physics equations can be modeled using differentiable functions in game engines.
Intelligent agents in games can be trained using techniques like machine
learning that are enabled by differentiation.

#### Simulations

Many simulation techniques for fluids and other physical processes are based on
approximate solutions to equations defined in terms of derivatives, like the
[Euler equations](https://en.wikipedia.org/wiki/Euler_equations_\(fluid_dynamics\))
and [Navier-Stokes](https://en.wikipedia.org/wiki/Navier–Stokes_equations).
Being able to differentiate functions is an important building block for
implementing algorithms to solve these equations.

#### Robotics

Control algorithms used in robotics and mechanical engineering rely on (often
higher-order) derivatives of functions that model the behavior of joints and
other physical systems. A language like Swift that can efficiently compute these
derivatives without incurring the unpredictable runtime overhead of garbage
collection may be well-placed to run aboard robots.

#### Rendering and ray tracing

Traditional rendering systems are black boxes that consume data structures with
scene geometry and produce images, but the physical processes they simulate are
made up of differentiable functions. Building a ray tracer out of differentiable
building blocks unlocks applications like inverse rendering (going from an image
to scene geometry). [[1]](https://github.com/BachiLi/redner)
[[2]](https://github.com/avik-pal/RayTracer.jl)

## Math introduction

### What is a derivative?

The derivative of a function `f` measures how quickly the function's output
changes when you make small changes to the function's input. The value of this
measurement depends on the input `x` that you start with, and we call the value
of the measurement starting at that input "the derivative of `f` at `x`.

For a single variable real function (a function with a single real input and a
single real output), the derivative of `f` at `x` can be summarized as a single
real number `f'(x)` such that `f(x + ε) ~= f(x) + f'(x) * ε`. In other words,
changing the input by a tiny amount `epsilon` changes the output by `f'(x) * ε`.

<p align="center">
  <img src="assets/0000-differentiable-programming/plot-linear.png">
  <br>
  <sub>
  <code>f(x) = x</code> changes by exactly <code>ε</code> whenever you change
  its input by <code>ε</code>, so its derivative is 1 everywhere.
  </sub>
</p>

<p align="center">
  <img src="assets/0000-differentiable-programming/plot-quadratic.png">
  <br>
  <sub>
  Near <code>x = 0</code>, <code>f(x) = x^2</code> changes very little when you
  change its input, so its derivative at <code>x = 0</code> is <code>0</code>
  (see orange line).
  <br>
  Near <code>x = 1</code>, <code>f(x) = x^2</code> changes by approximately
  <code>2*ε</code> when you change its input by <code>ε</code>, so its
  derivative at <code>x = 1</code> is <code>2</code> (see green line).
  <br>
  In general, the derivative of <code>f(x) = x^2</code> at <code>x</code> is
  <code>2*x</code>.
  </sub>
</p>

### Iterative optimization

Iterative optimization algorithms use derivatives to optimize functions (i.e.
find the inputs that minimize or maximize the output of the function). For
example, the simple "gradient descent" algorithm starts with an arbitrary input
`x` and uses the derivative of the function at `x` to determine whether it needs
to increase or decrease `x` to decrease the output of the function. Then it
mutates `x` slightly along the appropriate direction and repeats until the
output stops decreasing.

<p align="center">
  <img src="assets/0000-differentiable-programming/iterative-optimization.png">
</p>

### Derivatives of functions with arbitrary inputs

Real world programs deal with data more complicated than single real variables.
Fortunately, there are mathematical theories that extend derivatives to
functions with nearly arbitrary inputs and outputs.

Recall our original description of derivative: "The derivative of a function `f`
measures how quickly the function's output changes when you make small changes
to the function's input." This makes sense for arbitrary input and output types,
as long as we can describe small changes in them.

It is easy to describe small changes in nested structures of real numbers: they
are just small changes in all the components' real numbers. For example,
consider:

```swift
struct Point {
    var x, y: Float
}

struct PointPair {
    var p1, p2: Point
}
```

A small change in `Point` might be "add `0.01` to `x` and add `0.02` to y". A
small change in `PointPair` might be "add `0.01` to `p1.x` and add `0.01` to
`p2.x`".

We can define new types that capture the values of these small changes. We call
these types "tangent vectors", a term from math. For example:

```swift
extension Point {
    struct TangentVector {
        // `dx` and `dy` are small changes in `x` and `y`, respectively.
        var dx, dy: Float
    }
}

extension PointPair {
    struct TangentVector {
        // `dp1` and `dp2` are small changes in `p1` and `p2`, respectively.
        var dp1, dp2: Point.TangentVector
    }
}
```

In terms of these tangent vectors, the small changes that we described in words
above would be:

```swift
Point.TangentVector(dx: 0.01, dy: 0.02)

PointPair.TangentVector(
    p1: Point.TangentVector(dx: 0.01, dy: 0),
    p2: Point.TangentVector(dx: 0.01, dy: 0))
```

In terms of tangent vectors, the derivative of a function `f: (A) -> B` is a
function `df: (A, A.TangentVector) -> B.TangentVector`. In other words, `df`
takes a starting value of type `A` and a small change `A.TangentVector` and
tells you what the resulting small change in `B` is.

The gradient descent iterative optimization algorithm can run on any function
`f: (A) -> Float` as long as `A` is a type for which we can define a tangent
vector. It iteratively walks around different values of `A`, searching for a
value that minimizes the output of `f`.


## History of differentiation algorithms

There are three main algorithms for computing derivatives: numerical
differentiation, symbolic differentiation, and automatic differentiation.

### Numerical differentiation

Numerical differentiation is a technique for estimating derivatives of
mathematical functions using values of the functions. The simplest method uses
the
[difference quotient formula](https://en.wikipedia.org/wiki/Difference_quotient),
introduced in elementary calculus courses:

<p align="center">
  <img src="assets/0000-differentiable-programming/difference-quotient.png">
</p>

Numerical differentiation is easy to implement and generalizes to higher-order
derivatives. However, as an estimation approach, it is known to produce
inaccurate results, so it is rarely used when more accurate methods are
available.

### Symbolic differentiation

Symbolic differentiation is a technique for computing derivatives of math
expressions via symbolic manipulation, like differentiating an expression using
pen and paper in elementary calculus. This technique is used by computer algebra
systems like Mathematica, but it produces inefficient code when applied to
computer programs due to code bloat with common subexpressions.

### Automatic differentiation

Automatic differentiation (AD) is a technique for computing derivatives of
functions. Unlike symbolic differentiation, which operates on math expressions,
automatic differentiation operates on code.

Automatic differentiation leverages the chain rule of differentiation and the
ability to define temporary values in a program. There are two styles of
automatic differentiation in the traditional sense: forward-mode AD starts with
partial derivatives at inputs and ends by computing partial derivatives at
outputs, while reverse-mode automatic differentiation starts with partial
derivatives at outputs and ends by computing partial derivatives at inputs.

Mathematically, forward-mode AD corresponds to a fully-right association of the
chain rule of differentiation, and reverse-mode AD corresponds to a fully-left
association. Different associations of the chain rule produce the same result
but may differ in computational complexity†.

<p align="center">
  <img src="assets/0000-differentiable-programming/chain-rule-right-assoc.png" height=45 width=auto>
  <img src="assets/0000-differentiable-programming/chain-rule-left-assoc.png" height=45 width=auto>
  <br>
  <sub>
  Top: fully-right association of chain rule, starting from partial
  derivative of input; "forward-mode".
  <br>
  Bottom: fully-left association of chain rule, starting from output;
  "reverse-mode".
  </sub>
</p>

Both forward-mode AD and reverse-mode AD are well-explored. Forward-mode AD can
be implemented simply by overloading math operations to compute both original
values and derivatives. Traditionally, reverse-mode AD has been perceived as
being more complicated: implementations typically involve non-local program
transformation and/or mutable tape data structures, though recent research aims
to demystify the subject [[1]](https://arxiv.org/abs/1804.00746)
[[2]](https://arxiv.org/abs/1803.10228).

†: Finding the optimal association of the chain rule of differentiation is
analogous to the
[matrix chain multiplication](https://en.wikipedia.org/wiki/Matrix_chain_multiplication)
problem and can be solved in `O(n^3)` time. More efficient algorithms also
exist.

## Approaches to automatic differentiation

In practice, automatic differentiation is the most common differentiation
algorithm because it is precise and efficient. This section summarizes
approaches to automatic differentiation.

### Embedded domain-specific languages

A domain-specific language (DSL) is a language designed to solve problems for a
specific domain. Some DSLs are *external*: these are standalone languages with
their own syntax and semantics, like HTML (a markup language) and SQL (a
database query language). Other DSLs are *embedded* within a more general "host"
language: these DSLs leverage host language constructs and features to define
interesting behavior. Advantages of embedded DSLs include flexibility and
portability: embedded DSLs can be imported as a library. Examples of embedded
DSLs include React (a UI language embedded in JavaScript) and LINQ (a query
language embedded in C#).

One approach to differentiable programming is to define an embedded DSL for
differentiation *as a library*. This can be done via operator overloading: the
DSL can define a "dual number" type (representing a pair of a real number and
its derivative) and overload differentiable math operations to compute both
original values and derivative values.

```swift
struct RealWithDerivative<T: FloatingPoint> {
    var value: T
    var derivative: T = 0
}
extension RealWithDerivative {
    static func + (lhs: Self, rhs: Self) -> Self {
        RealWithDerivative(
            value: lhs.value + rhs.value,
            derivative: lhs.derivative + rhs.derivative)
    }
    static func * (lhs: Self, rhs: Self) -> Self {
        RealWithDerivative(
            value: lhs.value * rhs.value,
            derivative: lhs.derivative * rhs.value + lhs.value * rhs.derivative)
    }
}

var x = RealWithDerivative(value: 3, derivative: 1)
// Original:   x^2 + x^3 = 3^2 + 3^3 = 36.
// Derivative: 2x + 3x^2 = 2*3 + 3(3)^2 = 33.
var result = x*x + x*x*x
print(result)
// RealWithDerivative<Double>(value: 36.0, derivative: 33.0)
```

Such a DSL could be extended to be more useful. For example, the `Real` type
could be generalized to multidimensional arrays and more differentiable
operations could be added.

However, embedded DSLs have some limitations:

-   DSL functionality is often restricted to specific types and APIs. DSLs often
    use specialized abstractions rather than general ones for simplicity and to
    enable optimizations. For example, many machine learning frameworks are DSLs
    that support differentiation only for a particular multidimensional array
    type and only using a particular algorithm (reverse-mode automatic
    differentiation). Extending a differentiation DSL beyond these limitations
    is difficult and may require extra boilerplate: see below.

-   They typically involve some boilerplate. As a host language, Swift currently
    supports limited metaprogramming for reducing boilerplate code. For example,
    libraries cannot define automatic conformance derivation for library
    protocols (though Swift provides it for `Equatable`, `Hashable`, and
    `Codable`), so users must write boilerplate conformances for their custom
    types.

-   They are limited by the metaprogramming capabilities of the host language.
    It is not currently possible to define non-trivial code transformations
    (e.g. reverse-mode automatic differentiation) in a Swift library on Swift
    code. (Note: SwiftSyntax enables Swift AST transformations but has the extra
    indirection of parsing Swift code from a file - it is not possible to
    evaluate transformed Swift code from the same file without a general "eval"
    mechanism.) To cope with this, some DSLs require explicit program "graph"
    building and/or global mutable data structures to mimic the effects of code
    transformation, which obfuscate the original transformation semantics.

-   They may not work well with all host language constructs. Embedded DSLs only
    support a subset of the host language's features. In particular, some
    differentiation DSLs do not support native mutation (e.g. assigning to a
    `var`) or native control flow (e.g. `if` constructs) due to technical
    limitations, even though supporting them would be ideal.
    Restricting/diagnosing unsupported host language features (e.g. preventing
    DSL users from using `var` in Swift) is difficult or not possible.

-   Producing good diagnostics may be difficult or impossible. DSLs have limited
    access to source location information. When indirections like code
    transformations are involved, showing the appropriate source locations in
    diagnostic messages may be difficult. Without the aid of compiler utilities,
    statically detecting and diagnosing dataflow-based errors is not possible.

### Source code transformation tools

Source code transformation tools are another approach to differentiable
programming. Tool users write code, select various differentiation configuration
options (the name of the function-to-differentiate, the independent and
dependent variable, etc), and provide them to the tool. The tool analyzes the
input code and generates output code that computes derivatives according to the
options.

Historically, this is one of the oldest approaches for automatic
differentiation. Tools like
[Tapenade](https://www-sop.inria.fr/tropics/tapenade.html) and
[ADIC](https://www.mcs.anl.gov/research/projects/adic)/[ADIFOR](https://www.mcs.anl.gov/research/projects/adifor)
compute derivatives of Fortran and C code.

An advantage of source code transformation tools is that they are essentially
*static compilers*: they can perform static analyses on input code to generate
optimized derivative-computing output code. For example, Tapenade performs
["activity analysis"](https://www-sop.inria.fr/tropics/papers/supportCoursDA.pdf)
to determine variables that do not need a derivative and "TBR (to-be-recorded)
analysis" to remove unnecessary intermediate variables during differentiation.

However, these tools are not ideal for usability: users must interact with an
external GUI to specify inputs and they receive a textual program as output.
This external workflow is an extra indirection that takes users out of their
natural programming environment. Exposing the tool-provided differentiation
features within a language would be more ergonomic.

<p align="center">
  <img src="assets/0000-differentiable-programming/tapenade.png" height=500px>
  <br>
  <sub>
  Image of Tapenade web interface.
  <br>
  User specifies input program and configuration options.
  <br>
  Tapenade generates derivative-computing output program.
  </sub>
</p>

### First-class language support

Another class of differentiable programming approaches is by integrating the
differentiation semantics and code transformations into a programming language
to some degree. While there are no mainstream programming languages that support
differentiable programming, research systems like
[Stalingrad](http://www-bcl.cs.may.ie/~barak/papers/toplas-reverse.pdf) add
first-class differential operators (e.g. `grad`) into the language and the
reverse-mode automatic differentiation transformation into the compiler.

First-class language support for differentiation can reap the benefits of source
code transformation techniques (e.g. language coverage, performant derivative
code) without requiring programmers to use an external tool. Well-designed,
powerful differentiation primitives enable users to define their own custom
differentiation APIs that would otherwise not be possible in differentiation
libraries.

### Why bake differentiation into Swift?

First-class language support for differentiation will enable convenient,
extensible, and performant differentiable programming in Swift.

#### Maximal coverage of Swift language features

First-class support for differentiation in Swift enables differentiation to work
nicely with a maximal number of Swift language features, including mutation and
control flow. Users of differentiable programming do not need to write in a
restricted subset of Swift: just write normal code and use differentiation.

#### Extensibility

First-class language support enables an extensible differentiable programming
system.

Custom types can be extended to be differentiable with minimal boilerplate.
Custom derivative functions can be retroactively registered for existing
functions. Users can define custom differentiation APIs using the powerful
primitive operators defined in the standard library and supported by the type
system.

#### Static warnings and errors

Some functions perform non-differentiable operations (on the path from
parameters to result) and thus cannot be differentiated. Functions that do not
use their parameters to compute the result are technically differentiable, but
the derivative is trivially always zero.

With language support for differentiation, the compiler can identify these cases
statically via data flow analysis and produce a non-differentiability error or
warning. These diagnostics improve productivity and help users catch errors
ahead of time. Library-based differentiation approaches cannot generally provide
these diagnostics.

For details on static warnings and errors, see the "Static analysis" section in
the detailed design below.

#### The pursuit for user-defined code transformations

The key code transformation enabling differentiable programming is "derivative
code generation". Derivative code generation implements automatic
differentiation: given an "original function" to differentiate, a derivative
function is generated by replacing function applications in the original
function with corresponding derivative function applications. The algorithm is
described in detail in the
[Swift Differentiable Programming Implementation Overview document](http://bit.ly/swift-autodiff-internals).

Some languages provide the ability to define custom code transformations:

-   [Macros](https://en.wikipedia.org/wiki/Macro_\(computer_science\)) enable
    syntax-based code transformations at compile time. Hygienic macros (macro
    systems that avoid accidental variable capture) are available in a variety
    of languages, including Lisp, Julia, Rust, and Scala, to name a few. As an
    example: generated type-safe schema wrappers can implemented using
    [hygienic macros in Scala](https://meta.plasm.us/posts/2013/07/11/fake-type-providers-part-2).

-   Compiler plugin systems enable programmers to write plugins that extend the
    behavior of a compiler. Compiler plugins are more popular in bootstrapped
    languages, like
    [Haskell](https://ghc.gitlab.haskell.org/ghc/doc/users_guide/extending_ghc.html#compiler-plugins),
    [Rust](https://doc.rust-lang.org/1.1.0/book/compiler-plugins.html) and
    [Scala](https://docs.scala-lang.org/overviews/plugins/index.html), where the
    plugin can be written in the language itself. As an example: a
    continuation-passing-style code transformation can be implemented as a
    [compiler plugin in Scala](https://github.com/scala/scala-continuations).

One might make the case that derivative code generation for differentiation is
better implemented as a custom code transformation. While that may be true in
theory, Swift does not yet support custom code transformations in practice. This
proposal presents differentiable programming as a system of *high-level language
features and semantics*; derivative code generation is an implementation detail.
If a system for custom code transformations is added to Swift one day, it may be
possible to reimplement derivative code generation using that system without
changing the high-level differentiable programming features proposed here.

## Proposed solution

To push Swift's capabilities to the next level in numerics and machine learning,
we introduce differentiable programming as a new language feature, which
includes standard library APIs and small additive changes to the type system.

### The `Differentiable` protocol

`Differentiable` is a protocol defined in the standard library that generalizes
all data structures that can be a parameter or result of a differentiable
function. The compiler derives protocol requirement implementations when a
conformance is declared and when any implementation is missing.

```swift
extension Float: Differentiable {
    typealias TangentVector = Self
}
struct Perceptron: Differentiable {
    var weight: SIMD64<Float>
    var bias: Float
}
```

### The `@differentiable(reverse)` declaration attribute

The `@differentiable(reverse)` declaration attribute is an attribute that marks
function-like declarations (function declarations, initializers, properties, and
subscripts) as being differentiable.

```swift
@differentiable(reverse)
func cubed(_ x: Float) -> Float {
    x * x * x
}
extension Perceptron {
    @differentiable(reverse)
    func callAsFunction(_ input: SIMD64<Float>) -> Float {
        (weight * input).sum() + bias
    }
}
```

In [Differentiable Programming Manifesto], it is described that the
differentiable programming feature uses `@differentiable` without `(reverse)`.
However, we choose not to use `@differentiable` here because the initial set of
proposed feature do not include forward-mode differentiation. Adding `(reverse)`
makes room for future feature addition without ABI breakage.

### `@differentiable(reverse)` function types

Differentiable functions are first-class values, identified by a
`@differentiable(reverse)` attribute in the function type. A `@differentiable(reverse)` function
type is a subtype of its corresponding normal function type (i.e. without a
`@differentiable(reverse)` attribute) with an extended ABI, which stores extra
information that allows their values to be differentiated anywhere the function
is passed. A normal function can be implicitly converted to a `@differentiable(reverse)`
function with appropriate compile-time checks.

```swift
func addOne(_ x: Float) -> Float { x + 1 }
let _: @differentiable(reverse) (Float) -> Float = addOne
```

### `@derivative` attribute

The `@derivative` attribute is used for declaring custom derivative functions
for some other function declaration. This attribute can be used by libraries to
define differentiable functions that are "primitives", i.e. ones that the
compiler cannot differentiate automatically, or by the user to define special
behavior for debugging and performance tuning purposes.

The `Differentiation` library uses this attribute to define derivatives for math
functions, such as `expf(_:)` in the C standard library.

```swift
import Darwin // Or 'Glibc' on Linux

@usableFromInline
@derivative(of: expf)
func derivativeOfExpf(_ x: Float) -> (value: Float, pullback: (Float) -> Float) {
    let y = expf(x)
    return (value: y, pullback: { v in v * y })
}
```

### Differential operators

Standard library differentiation APIs that take `@differentiable(reverse)` functions and
return derivative functions or compute derivative values.

```swift
// In the standard library:
//     public func gradient<T, R: FloatingPoint>(
//       of body: @differentiable(reverse) (T) -> R
//     ) -> (T) -> T.TangentVector where R.TangentVector == R

func f(_ x: Float) -> Float {
    x * x
}
let dfdx = gradient(of: f)
dfdx(3) // 6
```


## Detailed design

### Differentiable data structures

Speaking in terms of elementary calculus, only functions are "differentiable":
only functions have derivatives and can be differentiated. In programming
languages, types are isomorphic to mathematical spaces, and functions are
isomorphic to mathematical functions over those spaces. Differentiability
depends heavily on the continuity and smoothness of points in a space (or values
of a type). For example, the `Int` type represents the space of integers, which
are discrete values, so functions over integers cannot be differentiated. In
general, when a type is said to be differentiable, it means that one can do
calculus with its values. As such, real numbers, real vector spaces, and complex
vector spaces are differentiable, but characters, strings, and integers are not.

For full flexibility and extensibility, a protocol is introduced in the Swift
standard library to generalize all data structures that can be a parameter or a
result of a differentiable function.

#### The `Differentiable` protocol

The `Differentiable` protocol defines operations and structures required for a
type to be differentiated.


```swift
public protocol Differentiable {
    /// A type that can be used to represent derivatives with respect to a
    /// value whose type is `Self`. Mathematically, this is equivalent to the
    /// tangent bundle of the differentiable manifold represented by the
    /// differentiable type.
    associatedtype TangentVector: Differentiable & AdditiveArithmetic
        where TangentVector == TangentVector.TangentVector

    /// Moves `self` along the given direction. In Riemannian geometry, this is
    /// equivalent to exponential map, which moves `self` on the geodesic
    /// surface along the given tangent vector.
    mutating func move(along direction: TangentVector)
    
    /// A closure that produces a zero tangent vector and does not capture `self`.
    ///
    /// In some cases, the zero tangent vector of `self` is equal to
    /// `TangentVector.zero`. In other cases, the zero tangent vector depends on
    /// information in `self`, such as shape for an n-dimensional array type.
    /// For differentiable programming, it is more memory-efficient to define a
    /// custom `zeroTangentVectorInitializer` property which returns a closure
    /// that captures and uses only the necessary information to create a zero
    /// tangent vector. For example:
    ///
    /// ```swift
    /// struct Vector {
    ///     var scalars: [Float]
    ///     var count: Int { scalars.count }
    ///     init(repeating repeatedElement: Float, count: Int) { ... }
    /// }
    /// 
    /// extension Vector: Differentiable {
    ///     typealias TangentVector = Vector
    ///
    ///     @noDerivative
    ///     var zeroTangentVectorInitializer: () -> TangentVector {
    ///         let count = self.count
    ///         return { TangentVector(repeating: 0, count: count) }
    ///     }
    /// }
    /// ```
    ///
    @noDerivative
    var zeroTangentVectorInitializer: () -> TangentVector { get }
}

extension Differentiable {
    /// A tangent vector such that `move(along: zeroTangentVector)` will not modify
    /// `self`.
    @noDerivative
    var zeroTangentVector: TangentVector { zeroTangentVectorInitializer() }
}
```

Specifically, `Differentiable` generalizes types to satisfy the following
requirements from real-world use cases: Functions over these types can be
differentiable. Besides types, a function's differentiability also depends on
the function's body. Values of these types can be updated based on derivative
values. For full flexibility, differentiable types should not be required to be
a vector space. For example, a differentiable neural network layer can store a
`Bool` flag in addition to differentiable parameters.

Intuitively, a `Differentiable`-conforming type allows one to do calculus with
its values. In elementary calculus, a derivative of a real-valued function at a
point is the slope of the tangent line at this point. The tangent line is the
best [linear approximation](https://en.wikipedia.org/wiki/Linear_approximation)
of the differentiated function near that input value. The same definition
applies to vector-valued functions when they are split into their coordinate
functions. The derivative of a vector-valued function at a certain point is
called a [tangent vector](https://en.wikipedia.org/wiki/Tangent_vector). Beyond
real numbers and vector spaces, there is a widely accepted mathematical
framework,
[differential geometry](https://en.wikipedia.org/wiki/Differential_geometry),
which generalizes calculus beyond Euclidean space. By bringing ideas from this
mathematical framework into the Swift standard library and the Swift compiler,
differentiable programming becomes more flexible and expressive than ever.

<p align="center">
  <img src="assets/0000-differentiable-programming/differentiable-manifolds.png" height=300px>
  <br>
  <sub>
  Image showing two differentiable manifolds: a sphere and a spheroid, from
  https://en.wikipedia.org/wiki/Pushforward_(differential).
  <br>
  If a map, φ, carries every point on manifold M to manifold N, then the
  pushforward of φ carries vectors in the tangent space at every point in M to
  a tangent space at every point in N.
  </sub>
</p>

Mathematically speaking, types that conform to `Differentiable` are considered
[smooth Riemannian manifolds](https://en.wikipedia.org/wiki/Riemannian_manifold).
When differentiating a function over these manifolds, a derivative value is a
vector in the [tangent bundle](https://en.wikipedia.org/wiki/Tangent_bundle) of
this manifold and has type `TangentVector`. The associated type `TangentVector`
is required to conform to `AdditiveArithmetic` because
[additive group](https://en.wikipedia.org/wiki/Additive_group) properties
[`zero`](https://developer.apple.com/documentation/swift/additivearithmetic/3126829-zero)
and
[`+(_:_:)`](https://developer.apple.com/documentation/swift/additivearithmetic/3126821)
are necessary for initializing and accumulating derivative values.

The `move(along:)` method is equivalent to the mathematical notion of
[exponential map](https://en.wikipedia.org/wiki/Exponential_map_\(Riemannian_geometry\)),
which takes a tangent vector (e.g. a derivative), and moves the value along the
direction specified by the tangent vector on the geodesic surface of the
manifold. In vector spaces where the tangent vector is of the same vector space
as the original differentiable space, `move(along:)` is equivalent to vector
addition. Mathematical optimization algorithms such as gradient descent will
make use of this method.

```swift
public extension Differentiable where Self == TangentVector {
    mutating func move(along direction: TangentVector) {
        self += direction
    }
}
```

The `zeroTangentVectorInitializer` property returns a closure that returns a
tangent vector such that calling `move(along:)` on the vector will not modify
`self`. A zero tangent vector is often used in the initialization of
mathematical optimization, where tangent vectors are initially zero and modified
iteratively. This property may be different from `TangentVector.zero` because
some tangent vectors depend on instance properties of `self`, e.g. the `count`
property in `Array`.

#### `Differentiable` conformances

Conforming a type to `Differentiable` tells Swift that changes in values of this
type can be differentiated, and makes functions over this type be compatible
with all differentiation APIs in the standard library. Floating point numeric
types and vector types, including
[`Float16`](https://developer.apple.com/documentation/swift/float16),
[`Float`](https://developer.apple.com/documentation/swift/float),
[`Double`](https://developer.apple.com/documentation/swift/double),
[`Float80`](https://developer.apple.com/documentation/swift/float80), and
[SIMD vector types](https://developer.apple.com/documentation/swift/swift_standard_library/numbers_and_basic_values/simd_vector_types),
are extended to conform to `Differentiable`, and their `TangentVector`s equal
themselves.

Besides numeric types, collections of numeric types are also powerful data
structures in differentiable programming. For example, the
[`Array`](https://developer.apple.com/documentation/swift/array) type in the
standard library [conforms to
`Differentiable`](https://github.com/apple/swift/blob/c224468653366119690aeb34f290843f3e5f2fd6/stdlib/public/core/Array.swift#L2052)
conditionally when the `Element` type conforms to `Differentiable`. This makes
it possible to differentiate functions over arrays, and makes it easy to express
dynamic differentiable algorithms. Similarly, other common container types in
the standard library such as
[`Optional`](https://developer.apple.com/documentation/swift/optional), and
[`Result`](https://developer.apple.com/documentation/swift/result) can also be
made differentiable via a conditional protocol conformance. We will pursue
adding these conformances in a follow-up proposal.

```swift
// struct Array<Element>
extension Array: Differentiable where Element: Differentiable {
    // Note: `Array.TangentVector` cannot be `Array` because `Array.+` is used for
    // concatenation and therefore cannot satisfy the `AdditiveArithmetic`
    // conformance constraint.
    public struct TangentVector: Differentiable, AdditiveArithmetic {
        public typealias TangentVector = Self
        @differentiable(reverse)
        public var elements: [Element.TangentVector]
        @differentiable(reverse)
        public init(_ elements: [Element.TangentVector]) { self.elements = elements }
        ...
    }

    public mutating func move(along direction: TangentVector) {
        for i in indices {
            self[i].move(along: Element.TangentVector(direction.elements[i]))
        }
    }

    @noDerivative
    public var zeroTangentVectorInitializer: () -> TangentVector {
        { [zeroInits = map(\.zeroTangentVectorInitializer)] in
            TangentVector(zeroInits.map { $0() })
        }
    }
}

// enum Optional<Wrapped>
extension Optional: Differentiable where Wrapped: Differentiable {
    public struct TangentVector: Differentiable, AdditiveArithmetic {
        public typealias TangentVector = Self
        @differentiable(reverse)
        public var value: Wrapped.TangentVector?
        @differentiable(reverse)
        public init(_ value: Wrapped.TangentVector?) { self.value = value }
        ...
    }

    public mutating func move(along direction: TangentVector) {
        if let value = direction.value {
            self?.move(along: value)
        }
    }

    @noDerivative
    public var zeroTangentVectorInitializer: () -> TangentVector {
        switch self {
        case nil:
            return { TangentVector(nil) }
        case let x?:
            return { [zeroTanInit = x.zeroTangentVectorInitializer] in
                TangentVector(zeroTanInit())
            }
        }
    }
}
```

#### Compiler-synthesized conformances

In numerics and machine learning, high-level data structures such as neural
network layers and models are formed from smaller components stored as
properties in structure types and class types. In order to use these types for
differentiation, one must extend these types to conform to the `Differentiable`
protocol. Luckily, this need not be done manually in most cases—the compiler
automatically synthesizes conformances when a `Differentiable` conformance is
declared.

##### Synthesis conditions

The compiler automatically synthesizes implementations of `Differentiable`
protocol requirements for struct and class types. For a type, conditions for the
synthesis are:

1. There is a conformance to `Differentiable` declared for the type, either in
   the original type declaration or in an extension.

2. The conformance must be declared in the same file.

Here is an example where the synthesis conditions are satisfied.

```swift
struct Model: Differentiable {
    var weight: SIMD4<Double>
    var bias: Double
    let metadata1: Float
    let metadata2: Float
    let usesBias: Bool
}
```

##### Default synthesis

The compiler synthesizes a nested `TangentVector` structure type that contains
the `TangentVector`s of all stored properties (terms and conditions apply) that
conform to `Differentiable`, which we call **differentiable variables**.

Mathematically, the synthesized implementation treats the data structure as a
product manifold of the manifolds each differentiable variable's type
represents. Differentiable variables' types are required to conform to
`Differentiable` because the synthesized implementation needs to access each
differentiable variable's type's `TangentVector` associated type and invoke each
differentiable variable's implementation of `move(along:)` and
`zeroTangentVectorInitializer`. Because the synthesized implementation needs to
invoke `move(along:)` on each differentiable variable, the differentiable
variables must have a `move(along:)` which satisfies the protocol requirement
and can be invoked on the property. That is, the property must be either a
variable (`var`) or a constant (`let`) with a non-`mutating` implementation of
the `move(along:)` protocol requirement.

The synthesized `TangentVector` has the same effective access level as the
original type declaration. Properties in the synthesized `TangentVector` have
the same effective access level as their corresponding original properties.

The synthesized `move(along:)` method calls `move(along:)` for each pair of a
differentiable variable and its corresponding property in `TangentVector`.

The synthesized `zeroTangentVectorInitializer` property returns a closure that
captures and calls each stored property's `zeroTangentVectorInitializer`
closure. When memberwise derivation is not possible (e.g. for custom
user-defined `TangentVector` types), `zeroTangentVectorInitializer` is
synthesized as a `{ TangentVector.zero }` closure.

```swift
struct Foo<T: Differentiable, U: Differentiable>: Differentiable {
    // `x` and `y` are the "differentiable variables".
    var x: T
    var y: U
    let customFlag: Bool

    // The compiler synthesizes:
    //
    //     struct TangentVector: Differentiable, AdditiveArithmetic {
    //         var x: T.TangentVector
    //         var y: U.TangentVector
    //     }
    //
    //     mutating func move(along direction: TangentVector) {
    //         x.move(along: direction.x)
    //         y.move(along: direction.y)
    //     }
    //
    //     var zeroTangentVectorInitializer: () -> TangentVector {
    //         { [xTanInit = x.zeroTangentVectorInitializer,
    //            yTanInit = y.zeroTangentVectorInitializer] in
    //             TangentVector(x: xTanInit(), y: yTanInit())
    //         }
    //     }
}
```

###### Opt out of synthesis for a stored property

The synthesized implementation of `Differentiable` protocol requirements already
excludes stored properties that are not differentiable variables, such as stored
properties that do not conform to `Differentiable` and `let`
properties that do not have a non-mutating `move(along:)`. In addition to this
behavior, we also introduce a `@noDerivative` declaration attribute, which can
be attached to properties that the programmer does not wish to include in the
synthesized `Differentiable` protocol requirement implementation.

When a stored property is marked with `@noDerivative` in a type that declares a
conformance to `Differentiable`, it will not be treated as a differentiable
variable regardless of whether it conforms to `Differentiable`. That is, the
synthesized implementation of protocol requirements will not include this
property.

```swift
struct Foo<T: Differentiable, U: Differentiable>: Differentiable {
    // `x` and `y` are the "differentiable variables".
    var x: T
    var y: U
    @noDerivative var customFlag: Bool
    @noDerivative let helperVariable: T
}
```

For clarity as to which stored properties are to be included for
differentiation, the compiler will recommend that all stored properties that
cannot be included as differentiable variables (due to either lacking a
conformance to `Differentiable` or being a non-`class`-bound `let` property) be
marked with `@noDerivative`. When a property is not included as a differentiable
variable and is not marked with `@noDerivative`, the compiler produces a warning
asking the user to make the exclusion explicit along with fix-it suggestions in
IDEs.

```swift
struct Foo<T: Differentiable, U: Differentiable>: Differentiable {
    // `x` and `y` are the "differentiable variables".
    var x: T
    var y: U
    var customFlag: Bool
    let helperVariable: T
}
```

```console
test.swift:5:4: warning: stored property 'customFlag' has no derivative because 'Bool' does not conform to 'Differentiable'
    var customFlag: Bool

test.swift:5:4: note: add a '@noDerivative' attribute to make it explicit
    var customFlag: Bool
    ^
    @noDerivative 
    
test.swift:6:4: warning: synthesis of the 'Differentiable.move(along:)' requirement for 'Foo' requires all stored properties not marked with `@noDerivative` to be mutable
    let helperVariable: T

test.swift:6:4: note: change 'let' to 'var' to make it mutable
    let helperVariable: T
    ^~~
    var

test.swift:6:4: note: add a '@noDerivative' attribute to make it explicit
    let helperVariable: T
    ^
    @noDerivative 
```

##### Shortcut synthesis

In certain cases, it is not ideal to keep `Self` and `TangentVector` as separate
types. A most obvious example of this is when all of the following conditions
are met: `Self` is declared to conform to `AdditiveArithmetic`. All stored
properties are declared to conform to `AdditiveArithmetic`. There are no
`@noDerivative` stored properties.

In these cases, the compiler will make `TangentVector` be a type alias for `Self`.
Method `move(along:)` will not be synthesized because a default implementation
already exists.

```swift
struct Point<T: Real>: Differentiable, AdditiveArithmetic {
    // `x` and `y` are the "differentiation properties".
    var x, y: T

    // The compiler synthesizes:
    //
    //     typealias TangentVector = Self
    //
    //     @noDerivative
    //     var zeroTangentVectorInitializer: () -> TangentVector {
    //         { [xTanInit = x.zeroTangentVectorInitializer,
    //            yTanInit = y.zeroTangentVectorInitializer] in
    //             TangentVector(x: xTanInit(), y: yTanInit())
    //         }
    //     }
}
```

### Differentiable function declarations

At the heart of a differentiable programming language is the ability to express
differentiable functions, from abstract manifold operations all the way down to
floating point addition. Because differentiable programming is a flexible and
extensible language feature in Swift, the compiler is agnostic of actual
mathematical operations—it does not have special knowledge of standard library
operators such as
[Float.+(_:_:)](https://developer.apple.com/documentation/swift/float/2894068),
nor does it distinguish between primitive operations and normal functions. A
function can be differentiated with respect to certain Differentiable-conforming
parameters if it satisfies one of the following requirements:

-   Base case: A derivative function for it with respect to those parameters
    exists in code.

-   Recursive case: All function calls, initializer calls, subscript accesses,
    property accesses, variable assignments along the path from those parameters
    to the result can be differentiated.

#### The `@differentiable(reverse)` declaration attribute

The `@differentiable(reverse)` declaration attribute can be used to mark function
declarations, initializers, properties, and subscripts as being differentiable.
When one of these entities is marked with `@differentiable(reverse)`, the compiler
attempts to differentiate it with respect to all parameters (including any
implicit `self` parameter) that conform to the `Differentiable` protocol. One
can specify explicit parameters via a `wrt:` clause, e.g. `@differentiable(reverse, wrt:
x)` and `@differentiable(reverse, wrt: (self, x))`. In generic algorithms, one can also
provide a `where`-clause to specify generic constraints for parameters or the
result to make the function differentiable only when the generic constraints are
satisfied, e.g. `@differentiable(reverse, wrt: x where Scalar: FloatingPoint)`.

```swift
@differentiable(reverse) // differentiable with respect to 'x'
func silly(_ x: Float, _ n: Int) -> Float {
    print("Running 'silly' on \(x) and \(n)!")
    return sin(cos(x))
}
```

Computed property getters behave like methods in that they accept exactly one
argument, `self`. If a computed property is marked with `@differentiable(reverse)`, the
compiler attempts to differentiate its getter with respect to `self`.
`@differentiable(reverse)` can also be applied to an explicit getter declaration.

```swift
extension Float {
    @differentiable(reverse)
    var reciprocal: Float {
        1 / self
    }
}
```

Among these language constructs, stored properties are the least method-like in
that they are stored values and cannot have a user-defined getter. However,
access to stored properties can be considered as a projection of `self`.
Therefore, stored properties can be marked `@differentiable(reverse)` and be
differentiated as a function as well. However, an explicit `@differentiable(reverse)` is
only necessary for public properties in public structs or classes to support
library evolution, and are implicitly synthesized by the compiler when the
parent type's `Differentiable` conformance is synthesized by the compiler (not
user-defined).

```swift
public struct Vector: Differentiable {
    @differentiable(reverse) // Okay, though the compiler has synthesized it.
    public var x, y: Float
}
```

#### Conformance and subclassing

Protocol requirements and class members can be made differentiable with a
`@differentiable(reverse)` attribute. Semantically, this means that this member is
guaranteed to be differentiable, and that any conformance implementation or
inheritance must maintain the differentiability.

##### Protocol dispatch

The `@differentiable(reverse)` attribute can be used on protocol requirements. A
`@differentiable(reverse)` protocol requirement requires that all conforming types
implement this requirement with a differentiable body with respect to the
specified parameters. Conforming implementations are not required to be marked
with `@differentiable(reverse)` attribute unless they are `public`.

```swift
public protocol Layer: Differentiable {
    associatedtype Input: Differentiable
    associatedtype Output: Differentiable
    @differentiable(reverse) // w.r.t. `input` and `self`
    func callAsFunction(_: Input) -> Output
}
struct Perceptron: Differentiable, Layer {
    var weight: SIMD4<Float>
    var bias: Float

    func callAsFunction(_ input: SIMD4<Float>) -> Float {
        (weight * input).sum() + b
    }
}
```

In a protocol hierarchy, one can override a differentiable protocol requirement
with a `@differentiable(reverse)` attribute that declares differentiability with respect
to more parameters.

```swift
public protocol Module: Differentiable {
    associatedtype Input
    associatedtype Output: Differentiable
    @differentiable(reverse, wrt: self)
    func callAsFunction(_: Input) -> Output
}

public protocol Layer: Module where Input: Differentiable {
    @differentiable(reverse, wrt: (self, input))
    func callAsFunction(_: Input) -> Output
}
```

In the example above, types that are declared to conform to `Layer` (the
protocol with a refined `callAsFunction(_:)` method) can omit the
`@differentiable(reverse, wrt: self)` attribute on the method implementation and use
`@differentiable(reverse, wrt: (self, input))` (or just `@differentiable(reverse)`) only.

`Differentiable` protocol requirements are not allowed to use a `where`-clause
in the `@differentiable(reverse)` attribute. This is to simplify the programming model
where protocol requirement overrides are more powerful.

##### Class dispatch

A differentiable non-final class method, property or subscript can be overridden
by a subclass implementation. The overriding implementation must be
`@differentiable(reverse)` if the original overridden declaration is marked with
`@differentiable(reverse)`. When a method/subscript call or a property access that is
dynamically dispatched is being differentiated, the derivative of the subclass
implementation will be used.

```swift
class Superclass {
    @differentiable(reverse)
    func foo(_ x: SIMD8<Float>) -> Float {
        x.sum()
    }
}

class Subclass: Superclass {
    @differentiable(reverse)
    override func foo(_ x: SIMD8<Float>) -> Float {
        (x * x).sum()
    }
}
```

### Make a function differentiable using `@derivative`

Any function that has `Differentiable`-conforming parameters and result can be
made differentiable by extending the function to have either an associated
derivative function. In other words, derivative functions provide
differentiability for other functions.

The `@derivative` attribute is used for marking a function as producing a custom
derivative for another function, hence making the other function differentiable.

A protocol requirement or class method/property/subscript can be made
differentiable via a derivative function defined in an extension. When a
protocol requirement is not marked with `@differentiable(reverse)` but has been
made differentiable by a `@derivative` declaration in a protocol extension, a
dispatched call to such a member can be differentiated, and the derivative is
always the one provided in the protocol extension.

#### Derivative functions

A derivative function has the same parameters as the original function, but
returns a pullback function in addition to the original value. Computing both
the original value and the pullback is the most efficient way for the pullback
closure to capture anything it needs from the original computation, and is
important for flexibility and performance.

A derivative function is expected to have the same effective access level or the
same linkage as the original function. That is, a derivative function and its
corresponding original function must have the same access level (either declared
or inferred from the declaration context) with the exception that when the
original function is `public`, the derivative function can either be
`@usableFromInline internal` or `public` (aka. having public linkage).

In the following example, the 32-bit floating point exponential function
[`expf(_:)`](https://en.cppreference.com/w/c/numeric/math/exp) is imported from
the C standard library. The derivative function marked with `@derivative` makes
`expf(_:)` a differentiable function.

```swift
import Glibc

@usableFromInline
@derivative(of: expf)
func derivativeOfExpf(_ x: Float) -> (value: Float, pullback: (Float) -> Float) {
    let y = expf(x)
    return (value: y, pullback: { v in v * y })
}
```

When one declares a derivative function for an existing function, the derivative
function defined will be rarely ever used because it is already associated with
the original function and can be obtained by using a differential operator such
as `valueWithPullback(at:in:)`. Therefore, a possible future direction is to
allow functions to be declared with an anonymous identifier, such as `_`.
However, this is out of the scope of this proposal.

##### Typing rules

A function declaration does not have a fixed derivative type. This is because
there can be multiple derivative functions that differentiate the original
function differently, e.g. differentiating with respect to different parameters,
differentiating with different generic constraints, etc.

Given an original function declaration, a derivative function's type is
determined from the following configurations:
- Parameters to differentiate with respect to, aka. differentiability
  parameters.
- Additional generic constraints that make the original function differentiable.

The type of the derivative function under such configurations is a function that
takes the original function's parameters and returns a tuple of an original
result (labeled `value`) and a differential (labeled `differential`). The
pullback function takes the `TangentVector` nested type of the original
function's result type, and returns `TangentVector` nested types of all of the
types of the original function's parameters to differentiate with respect to.

###### Differentiability parameters

The `@derivative` attribute accepts a `wrt:` argument which specifies the
differentiability parameters. If `wrt:` is not specified, the derivative
function should be differentiating the original function with respect to all of
its parameters, hence producing a differential that takes all of the original
function's parameter types' `TangentVector` types. A `wrt:` argument in
`@derivative` attributes can be a parameter name, a parameter index, or a tuple
of multiple parameter names or indices. All differentiability parameters must
have a type that conforms to `Differentiable`.

A derivative function's argument labels must match those of the original
function. Its parameter names do not have to match those of the original
function. However, a `wrt:` argument in a `@derivative` attribute, when
referring to parameters by names, must use parameter names in the derivative
function.

```swift
func foo<T: Differentiable>(_ x: T, _ y: T, _ z: T) -> T { ... }

// Derivative with respect to all parameters.
@derivative(of: foo)
func derivativeOfFoo<T: Differentiable>(_ x: T, _ y: T, _ z: T) -> (
    value: T, 
    pullback: (T.TangentVector) -> (T.TangentVector, T.TangentVector, T.TangentVector)
) {
    ...
}

// Derivative with respect to `x`.
@derivative(of: foo, wrt: x)
func derivativeOfFoo<T: Differentiable>(_ x: T, _ y: T, _ z: T) -> (
    value: T, 
    pullback: (T.TangentVector) -> T.TangentVector
) {
    ...
}

// Derivative with respect to `x` and `z`.
@derivative(of: foo, wrt: (x, z))
func derivativeOfFoo<T: Differentiable>(_ x: T, _ y: T, _ z: T) -> (
    value: T, 
    pullback: (T.TangentVector) -> (T.TangentVector, T.TangentVector)
) {
    ...
}
```

One concrete example is `sinf(_:)` from the C standard library. It can be made
differentiable by defining a derivative retroactively.

```swift
#if canImport(Darwin)
import func Darwin.sinf
#else
import func Glibc.sinf
#endif

// Imported:
//     public func sinf(Float) -> Float

@derivative(of: sinf)
public func derivativeOfSinf(_ x: Float) -> (value: Float, pullback: (Float) -> Float) {
    (value: sinf(x), pullback: { v in cosf(x) * v })
}
```

###### Differentiability generic requirements

A derivative function can have additional generic constraints, called
_differentiability generic requirements_. Differentiability generic requirements
usually serve the purpose of making generic parameter types conform to
`Differentiable`.

Differentiability generic requirements are functionally equivalent to the
`where` clause in `@differentiable(reverse)` attributes.

```swift
func foo<T, U, V>(_ x: T, _ y: U, _ z: V) -> W { ... }

// Derivative with respect to `x` and `z`, requiring that `T` and `V` to conform
// to `Differentiable`.
@derivative(of: foo, wrt: (x, z))
func foo<T: Differentiable, U, V: Differentiable>(
    _ x: T, _ y: U, _ z: V
) -> (
    value: W, 
    pullback: (W.TangentVector) -> (T.TangentVector, V.TangentVector)
) {
    ...
}
```

#### Access control

The derivative function must have the same access level as the original
function. Customizing the access level of a function's differentiability is
technically possible, but it is out of scope for this proposal.

```swift
// File A.swift:
internal func foo(_ x: Float) -> Float {
    x * x
}
let dfdx_A = gradient(at: 3, in: foo)
// dfdx_A ==> 6

// File B.swift:
@derivative(of: foo)
func derivativeOfFoo(_ x: Float) -> (value: Float, pullback: (Float) -> Float) {
    (value: foo(x), pullback: { _ in 42 })
}
let dfdx_B = gradient(at: 3, in: foo)
// dfdx_B ==> 42

// File C.swift:
let dfdx_C = gradient(at: 3, in: foo)
// dfdx_C ==> 6
```

### Differentiable function types

Differentiability is a fundamental mathematical concept that applies not only to
declarations of functions, initializers, subscripts, and properties, but also to
function types. In Swift, functions are first-class values of function types
that can be passed around, applied, or converted. Because an important part of
differentiable programming is to be able to define [differential
operators](https://en.wikipedia.org/wiki/Differential_operator) and custom
algorithms on differentiable functions, we extend Swift's type system to be able
to express differentiable functions as first-class values.

A differentiable function type is a special function type that has a different
runtime representation than a normal function type, and is a subtype of a
non-differentiable function type with the same parameter types and result type.

#### Function subtyping and runtime representation

Subtyping of function types already exists in Swift and is primarily used for
representing different foreign calling conventions for language
interoperability. Function types and function pointer types in C, e.g.
`int(*)(int)`, are imported to Swift as function types with a `@convention(c)`
attribute, e.g. `@convention(c) (Int) -> Int`, with all parameter types and
return types converted to the corresponding Swift ones.

These function types are also subtypes of a function type with the same
parameter types and result types but without the `@convention(c)` attribute. For
example, you can implicitly convert a `@convention(c)` function value to a Swift
function value and use it directly as an argument to higher-order functions such
as
[`map(_:)`](https://developer.apple.com/documentation/swift/array/3017522-map).

```swift
// In a C file:
int addOne(int x) { return x + 1; }
int (*addOneFunctionPointer)(int) = addOne;
// Swift equivalent:
//   let addOneFunctionPointer: (Int) -> Int = addOne

// In a Swift file that imports the C file:
// Global variable `addOneFunctionPointer` imported as `@convention(c) (Int) -> Int`.
[1, 2, 3].map(addOneFunctionPointer) // [2, 3, 4]
```

In differentiable programming, differentiable function types contain more
information than its non-differentiable counterparts. A differentiable function
contains the original function pointer so that it can be efficiently converted
to or called like the original function type. It also contains a derivative
function that will be called when this function is being differentiated. All of
these functions share the same captured variables, but may have different
generic specialization information in the closure context, because derivatives
can be conditionally available.

#### The `@differentiable(reverse)` function type attribute

A `@differentiable(reverse)` attribute on a function type specifies the
function's differentiability, just like `@differentiable(reverse)` on function
declarations.

`@differentiable(reverse)` requires the enclosing function type to have differentiable
parameters and results. Each parameter and result must conform to the
`Differentiable` protocol unless marked `@noDerivative`.

#### Type conversion

The subtyping relation between `@differentiable(reverse)`, and
non-`@differentiable(reverse)` function types allow functions of different types to be
conditionally convertible to each other. Such conversions do not always succeed:
Conversion from a function declaration (`func`) to a `@differentiable(reverse)` function
value succeeds if and only if the function can be differentiated. Conversion
from a `@differentiable(reverse)` function value to a non-`@differentiable(reverse)` function
value always succeeds. Conversion from a non-`@differentiable(reverse)` function value to
a `@differentiable(reverse)` function value always fails, because the function's body is
opaque to the compiler.

##### Coercing function declarations into `@differentiable(reverse)` function values

A function declaration can be implicitly coerced into a `@differentiable(reverse)`
function value, when there is a contextual `@differentiable(reverse)` function type. Such
conversions succeed either if the function declaration has been marked with a
`@differentiable(reverse)` declaration attribute, or if the function declaration is
defined in the same module and the function can be differentiated as if it were
marked with `@differentiable(reverse)`. When neither of these conditions are met, the
function cannot be differentiated, and thus cannot be converted to a
`@differentiable(reverse)` function value, in which case the compiler will produce an
error.

```swift
func addOne(_ x: Float) -> Float { x + 1 }
let _: @differentiable(reverse) (Float) -> Float = addOne // Okay!

let _: @differentiable(reverse) (Float) -> Float = coshf(_:)
// Error: `coshf(_:)` is from a different module and has not been marked with
// `@differentiable(reverse)`.

func mySin(_ x: Float) -> Float { sin(x) * 2 }
let _: @differentiable(reverse) (Float) -> Float = mySin // Okay!

func addOneViaInt(_ x: Float) -> Float { Float(Int(x) + 1) }
let _: @differentiable(reverse) (Float) -> Float = addOneViaInt
// Error: When differentiating `addOneViaInt(_:)`, `Int(x)` is not differentiable.
```

##### Upcasting to non-`@differentiable(reverse)` functions

As shown in the [function subtyping and runtime
representation](#function-subtyping-and-runtime-representation) subsection, a
`@differentiable(reverse)` function value's runtime representation contains the original
function along with extra information that allows the function to be
differentiated. A `@differentiable(reverse)` function value can be called like a
non-`@differentiable(reverse)` function.
```swift
func addOne(_ x: Float) -> Float { x + 1 }
let f1: @differentiable(reverse) (Float) -> Float = f0
let f2: (Float) -> Float = f1
```

A `@differentiable(reverse)` function can also be converted to a function which is
identical except that more of its parameters are marked with `@noDerivative`.

```swift
func addOne(_ x: Float) -> Float { x + 1 }
let f0: @differentiable(reverse) (Float, Float, Float) -> Float = addOne
let f1: @differentiable(reverse) (@noDerivative Float, Float, Float) -> Float = f0
let f2: @differentiable(reverse) (@noDerivative Float, Float, @noDerivative Float) -> Float = f1
```

#### Non-differentiable parameters

Like function declarations with a `@differentiable(reverse)` attribute,
differentiable function values can also be differentiable with respect to a
subset of parameters. This is expressed as part of type information, in
`@differentiable(reverse)` function types, using a `@noDerivative` attribute at
each parameter that is not being differentiated with respect to.

By default, all parameters are being differentiated with respect to. When a
`@noDerivative` attribute is specified for a parameter in a
`@differentiable(reverse)` function type, values of this function type are not
differentiable with respect to the parameter.

```swift
let f0: @differentiable(reverse) (Float, Float) -> Float = { $0 * $1 }
let f3: @differentiable(reverse) (@noDerivative Int, Float, @noDerivative Int) -> Float = {
  $0 ? Float($1) + $2 : 0
}
```

Differentiability of parameters in a function type is important for type
conversions and is part of the subtyping rule: Any `@differentiable(reverse)`
function type is a subtype of the same function type with more `@noDerivative`
parameters than there originally are.

```swift
let f0: @differentiable(reverse) (Float, Float) -> Float = { $0 * $1 }
_ = f0 as @differentiable(reverse) (Float, @noDerivative Float) -> Float
_ = f0 as @differentiable(reverse) (@noDerivative Float, Float) -> Float
_ = f0 as @differentiable(reverse) (@noDerivative Float, @noDerivative Float) -> Float
```

### Differential operators

The `Differentiation` module will provide APIs which developers can use to
obtain gradient functions, gradient vectors, and pullback closures, along with
efficiently-computed original results from a given `@differentiable(reverse)`
closure. These APIs are called "differential opeators".

#### `gradient(of:)`

`gradient(of:)` is a higher-order function which behaves exactly like the 𝛁
([Del](https://en.wikipedia.org/wiki/Del)) operator in mathematics. It takes a
differentiable closure that returns a scalar and its gradient function, i.e. a
closure which accepts the same arguments as the input closure but returns
gradient vectors with respect to the input closure's parameter.

```swift
/// Returns the gradient function of the given closure with respect to the argument.
/// - Parameter:
///   - body: A closure whose derivative function will be evaluated.
/// - Returns: A gradient vector with respect to `x`.
func gradient<T: Differentiable, R: FloatingPoint & Differentiable>(
    of body: @escaping @differentiable(reverse) (T) -> R
) -> (T) -> T.TangentVector where R.TangentVector: FloatingPoint
```

#### `gradient(at:in:)`

`gradient(at:in:)` is the "uncurried" form of `gradient(of:)`. It takes a value
and a differentiable closure that returns a scalar, and evalutes the closure's
gradient function on the value.

```swift
/// Returns the gradient vector with respect to the argument by evaluating the
/// provided closure's derivative at the argument.
/// - Parameter:
///   - x: An argument to be passed to `body`.
///   - body: A closure whose derivative function will be evaluated.
/// - Returns: A gradient vector with respect to `x`.
func gradient<T: Differentiable, R: FloatingPoint & Differentiable>(
    at x: T, in body: @differentiable(reverse) (T) -> R
) -> T.TangentVector where R.TangentVector: FloatingPoint
```

The call sites of this API read as if the call is feeding an argument into the
trailing closure, getting back a gradient vector. This API is consistent with
developers' mental model on taking the gradient of algorithms, and therefore
will be the most commonly used API. For example, a deep learning model's
training loop may look like the following.

```swift
for _ in 0..<1000 {
    // Differentiate the loss with respect to the model `classifier` itself, producing a
    // tangent vector `modelGradient` that represents partial derivatives with respect to
    // all trainable model parameters in the model.
    let modelGradient = gradient(at: classifier) { classifier in
        let prediction = classifier(x)
        let loss = softmaxCrossEntropy(logits: prediction, labels: y)
        print("Loss: \(loss)")
        return loss
    }
    optimizer.performStep(for: model, along: modelGradient)
}
```

#### `valueWithGradient(at:in:)`

Sometimes the developer needs to obtain both the original result and the
gradient vector. While it is possible for the developer to call the
differentiable closure and `gradient(at:in:)` separately, it would lead to
significant recomputation overhead, because computing the gradient vector of a
differentiable closure at a value will already compute the closure's original
result. `valueWithGradient(at:in:)` is an API for efficiently computing both the
original result and the gradient vector.

```swift
/// Returns the result and gradient vector with respect to the argument by evaluating the
/// provided closure's derivative at the argument.
/// - Parameter:
///   - x: An argument to be passed to `body`.
///   - body: A closure whose derivative function will be evaluated.
/// - Returns: The result of `body` evaluated on `x`, equivalent to `body(x)`, and
///   a gradient vector with respect to `x`.
func valueWithGradient<T: Differentiable, R: FloatingPoint & Differentiable>(
    at x: T, in body: @differentiable(reverse) (T) -> R
) -> (value: R, gradient: T.TangentVector) where R.TangentVector: FloatingPoint
```

```swift
// Example: Want both the result and the gradient of `foo(x)`.
func foo(_ x: Double) -> Double {
    tanh(tanh(exp(x)))
}
let x = 2.0

// Slow way:
let y = foo(x)
let dydx = gradient(at: x, in: foo)

// Efficient way:
let (y, dydx) = valueWithGradient(at: x, in: foo)
```

#### `valueWithPullback(at:in:)`

`valueWithPullback(at:in:)` is the most general form of differential operator
for reverse-mode automatic differentiation. Unlike `valueWithGradient(at:in:)`
which directly computes the gradient vector, `valueWithPullback(at:in:)` returns
a pullback closure that represents a linear approximation of the input closure
at the given value. This formulation corresponds exactly to derivative functions
that are defined with `@derivative`, and enables the most flexibility and
composability. In fact, all other differential operators discussed above are
implemented in terms of `valueWithPullback(at:in:)`.

```swift
/// Returns the result and pullback closure by evaluating the provided closure's
/// derivative at the argument.
/// - Parameter:
///   - x: An argument to be passed to `body`.
///   - body: A closure whose derivative function will be evaluated.
/// - Returns: The result of `body` evaluated on `x`, equivalent to `body(x)`, and
///   a pullback closure, which represents a transposed linear combination that
///   approximates `body` at `x`. When evaluated on a tangent vector, `pullback` evaluates
///   the linear comibination on the tangent vector and returns a gradient vector with
///   respect to `x`.
func valueWithPullback<T: Differentiable, R: Differentiable>(
    at x: T, in body: @differentiable(reverse) (T) -> R
) -> (value: R, pullback: (R.TangentVector) -> T.TangentVector)
```

### Static analysis

Differentiable programming in Swift aims to provide the best static compiler
diagnostics to help users catch mistakes. Beyond error diagnostics, the compiler
and the standard library are equipped with static analyses and marker APIs that
help the user write differentiable code with explicit annotations about
non-obvious non-differentiable cases.

#### Cross-module opacity

Swift libraries are distributed as
[modules](https://docs.swift.org/swift-book/LanguageGuide/AccessControl.html),
which provide an API and an opaque binary format for client code to use. By
importing a library, we can compute derivatives of functions that have been
marked with `@differentiable(reverse)` or that have been provided with a
derivative function, but not of functions that have not been marked this way
without defining a custom derivative for it. For example, if we try to
differentiate [`sinf(_:)`](https://en.cppreference.com/w/c/numeric/math/sin)
with the `gradient(at:in:)` API, the compiler will produce error messages at
compile time instead of producing zero derivatives.

```swift
let y = derivative(at: 1.0) { x in
    sinf(x)
}
```

```console
test.swift:4:5: error: expression is not differentiable
    sinf(x)
    ^
test.swift:4:5: note: cannot differentiate functions that have not been marked '@differentiable(reverse)' and that are defined in other modules
    sinf(x)
    ^
```

#### Non-differentiable type conversions

Calling functions that convert values to non-differentiable types and convert
them back makes the function no longer differentiable. The compiler is able to
detect these cases and provide error messages.

```swift
let d = derivative(at: 1.0) { x in
    Double(Int(x)) + 2
}
```

```console
test.swift:1:27: error: function is not differentiable
let y = derivative(at: 1.0) { x in
                            ^~~~~~
test.swift:2:12: note: cannot differentiate through a non-differentiable result; do you want to add 'withoutDerivative(at:)'?
    Double(Int(x)) + 2
           ^
```

#### Accidental data flow mistakes

Even when there are no obvious non-differentiable operations on the path from
parameters to the result (like non-differentiable type conversions), it is still
possible to mistype a variable and cause numerical computation to be incorrect.
As such, the compiler is able to leverage dependency analysis to determine
whether the derivative is always zero and warns the user.

```swift
let grad = gradient(at: 1.0) { x in
    3.0.squareRoot()
}
```

```console
test.swift:2:4: warning: result does not depend on differentiation arguments and will always have a zero derivative
    3.0.squareRoot()
    ^
test.swift:2:4: note: add 'withoutDerivative(at:)' to silence the warning if zero derivatives are intentional
    3.0.squareRoot()
    ^
    withoutDerivative(at:  )
```

## Source compatibility

This feature does not change any existing APIs. While the addition of
`@differentiable(reverse)` function types changes the function implicit
conversion rules in the type checker, the relevent code paths are only triggered
when a `@differentiable(reverse)` function type is involved in a contextual
type.

## Effect on ABI stability

The ABI changes proposed is purely additive. Protocols with requirements marked
with `@differentiable(reverse)` will contain an extra entry storing its
corresponding derivative function, provided by conforming types. Similarly,
`@differentiable(reverse)` is a new function representation that represents a
bundle of two functions, the original function and the derivative function.

## Effect on API resilience

This feature adds the [`Differentiable` protocol](#differentiable-protocol) and
[differential operators](#differential-operators) to the standard library as
public APIs. They introduce additions to the standard library.

### `Differentiable` protocol

The `Differentiable` protocol contains all necessary requirements for a type to
be differentiated. Without breaking API, it will be possible to add extensions
to the `Differentiable` protocol and add new requirements with default
implementations.

### Differential operators

Differential operators (e.g. `derivative(of:)` and `gradient(of:)`) are added to
the standard library as lightweight top-level higher-order functions. These APIs
can be renamed or moved under some namespace without breaking ABI.

## Alternatives considered

### Not support differentiable programming

We believe first-class differentiable programming is a big step towards making
Swift a real contender in the numerical computing and machine learning
landscape. Differentiable programming will enable intelligent applications,
machine learning models, scientific experiments, physical simulations, and more.

### Use another language or framework for differentiable programming

Dynamic languages, like Python and Julia, have established library support for
differentiable programming. While it is possible to interoperate with these
libraries via Swift, we feel that first-class differentiable programming in
Swift is leaps ahead in expressivity, usability, and safety.

### Other approaches to differentiable programming

See
["Approaches to automatic differentiation"](#approaches-to-automatic-differentiation)
above for an overview and comparison of automatic differentiation approaches.
First-class language support for differentiation will enable convenient,
extensible, and performant differentiable programming in Swift - more so than
library-based approaches.

## Acknowledgements

The development of this feature started in early 2018 as part of the [Swift for
TensorFlow](https://www.tensorflow.org/swift) project and has been pioneered by
engineers from Google. The authors would like to thank everybody involved. See
the
[Acknowledgements](https://github.com/apple/swift/blob/main/docs/DifferentiableProgramming.md#acknowledgements)
section of the manifesto.

<!-- Links -->

[Richard Wei]: https://github.com/rxwei
[Dan Zheng]: https://github.com/dan-zheng
[Marc Rasi]: https://github.com/marcrasi
[Bart Chrzaszcz]: https://github.com/bartchr808
[Aleksandr Efremov]: https://github.com/efremale

[swift-numerics]: https://github.com/apple/swift-numerics
[SE-0229]: https://github.com/apple/swift-evolution/blob/master/proposals/0229-simd.md
[SE-0233]: https://github.com/apple/swift-evolution/blob/master/proposals/0233-additive-arithmetic-protocol.md
[SE-0246]: https://github.com/apple/swift-evolution/blob/master/proposals/0246-mathable.md
[SE-0251]: https://github.com/apple/swift-evolution/blob/master/proposals/0251-simd-additions.md

[Differentiable Programming Manifesto]: https://github.com/apple/swift/blob/main/docs/DifferentiableProgramming.md
