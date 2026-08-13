if (NOT MINGW)
    duckdb_extension_load(avro
            LOAD_TESTS
            GIT_URL https://github.com/duckdb/duckdb-avro
            # Iceberg manifests use Avro decimal logical types. Keep this in
            # sync with duckdb-iceberg's extension_config.cmake so a Vane
            # package does not silently link an older BLOB-only Avro reader.
            GIT_TAG 7f423d69709045e38f8431b3470e0395fce1a595
    )
endif()
