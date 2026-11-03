rule "Sandbox denied file read"
condition:
    any read in sandbox.file_read where
        read.payload.allowed == false and sandbox.state.mode == "enforce"
