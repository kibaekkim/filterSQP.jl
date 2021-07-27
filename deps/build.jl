using BinaryProvider

products = [
    LibraryProduct(prefix, ["libfilter"], :libfilter),
    ExecutableProduct(prefix, "filter", :amplexe),
]

function update_product(product::LibraryProduct, library_path, binary_path)
    LibraryProduct(library_path, product.libnames, product.variable_name)
end

function update_product(product::ExecutableProduct, library_path, binary_path)
    ExecutableProduct(joinpath(binary_path, basename(product.path)), product.variable_name)
end

if haskey(ENV,"JULIA_FILTERSQP_LIBRARY_PATH") && haskey(ENV,"JULIA_FILTERSQP_EXECUTABLE_PATH")
    custom_products = [update_product(product, ENV["JULIA_FILTERSQP_LIBRARY_PATH"], ENV["JULIA_FILTERSQP_EXECUTABLE_PATH"]) for product in products]
    if all(satisfied(p; verbose=verbose) for p in custom_products)
        products = custom_products
    else
        error("Could not install custom libraries from $(ENV["JULIA_FILTERSQP_LIBRARY_PATH"]) and $(ENV["JULIA_FILTERSQP_EXECUTABLE_PATH"]).\nTo fall back to BinaryProvider call delete!(ENV,\"JULIA_FILTERSQP_LIBRARY_PATH\");delete!(ENV,\"JULIA_FILTERSQP_EXECUTABLE_PATH\") and run build again.")
    end
end

# Write out a deps.jl file that will contain mappings for our products
write_deps_file(joinpath(@__DIR__, "deps.jl"), products, verbose=verbose)
