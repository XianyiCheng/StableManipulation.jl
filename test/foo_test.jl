# [test/foo_test.jl]
@testset "Foo test" begin
    v = StableManipulation.foo(10,5)
    @test v[1] == 10
    @test v[2] == 5
    @test eltype(v) == Int
    v = StableManipulation.foo(10.0, 5)
    @test v[1] == 10
    @test v[2] == 5
    @test eltype(v) == Float64
end