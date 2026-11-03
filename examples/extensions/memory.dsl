rule "Tool A committed persistent memory"
condition:
    any write in memory.write_committed where write.payload.source_tool == "tool_a"
