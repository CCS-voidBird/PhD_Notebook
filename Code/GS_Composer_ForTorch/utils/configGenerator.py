import argparse
import json
import yaml
import configparser
import os
from typing import Dict, Any, Optional, Union, List, Tuple


def generate_config_from_parser(
    parser: argparse.ArgumentParser,
    output_file: str,
    format: str = "yaml",
    include_defaults: bool = True,
    include_help: bool = True,
    include_choices: bool = True,
    include_required: bool = True,
    group_by_groups: bool = True
) -> Dict[str, Any]:
    """
    Generate a configuration file from an argparse ArgumentParser object.
    
    Args:
        parser: The ArgumentParser object to extract configuration from
        output_file: Path to save the configuration file
        format: Format of the config file ('yaml', 'json', or 'ini')
        include_defaults: Whether to include default values in the config
        include_help: Whether to include help text as comments
        include_choices: Whether to include available choices as comments
        include_required: Whether to include required status as comments
        group_by_groups: Whether to organize by argument groups
        
    Returns:
        Dictionary representation of the configuration
    """
    # Extract argument information from parser
    config_data = _extract_parser_info(
        parser, 
        include_defaults, 
        include_help,
        include_choices,
        include_required,
        group_by_groups
    )
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Write to specified format
    if format.lower() == "yaml":
        _write_yaml_config(config_data, output_file, 
                          include_help, include_choices, include_required)
    elif format.lower() == "json":
        _write_json_config(config_data, output_file)
    elif format.lower() == "ini":
        _write_ini_config(config_data, output_file, 
                         include_help, include_choices, include_required)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'yaml', 'json', or 'ini'.")
    
    print(f"Configuration file generated at: {output_file}")
    return config_data


def _extract_parser_info(
    parser: argparse.ArgumentParser,
    include_defaults: bool,
    include_help: bool,
    include_choices: bool,
    include_required: bool,
    group_by_groups: bool
) -> Dict[str, Any]:
    """Extract all relevant information from the parser object."""
    config_data = {}
    
    # Get all action groups
    action_groups = []
    
    # Add the default group (unnamed)
    if hasattr(parser, '_action_groups'):
        action_groups = parser._action_groups
    elif hasattr(parser, '_actions'):
        # Create a dummy group with all actions
        class DummyGroup:
            def __init__(self, actions, title):
                self._actions = actions
                self.title = title
        
        action_groups = [DummyGroup(parser._actions, "Default")]
    
    # Process each group
    for group in action_groups:
        
        group_title = group.title if hasattr(group, 'title') else "Default"
        
        # Skip the "positional arguments" and "optional arguments" default groups
        if group_title in ["positional arguments", "optional arguments"]:
            print(group.title)
            group_title = "Default"
        
        # Create a section for this group if needed
        if group_by_groups and group_title != "Default":
            if group_title not in config_data:
                config_data[group_title] = {}
            section = config_data[group_title]
        else:
            section = config_data
        
        # Process each action in the group
        for action in group._actions:
            # Skip help action
            if action.dest == 'help':
                continue
                
            # Get the argument name
            arg_name = _get_arg_name(action)
            if not arg_name:
                continue
                
            # Get default value if available
            default_value = action.default if include_defaults else None
            
            # Skip if default is suppressed or None and not including defaults
            if default_value is argparse.SUPPRESS:
                continue
                
            # Handle different action types
            if isinstance(action, argparse._StoreAction):
                if action.choices is not None and default_value is None:
                    default_value = action.choices[0] if action.choices else None
                    
                metadata = {}
                if include_help and action.help:
                    metadata['help'] = action.help
                if include_choices and action.choices:
                    metadata['choices'] = list(action.choices)
                if include_required:
                    metadata['required'] = action.required

                # Add the argument to the config
                section[arg_name] = {
                    'value': default_value,
                    'metadata': metadata
                }
                
            elif isinstance(action, argparse._StoreTrueAction) or isinstance(action, argparse._StoreFalseAction):
                is_true_action = isinstance(action, argparse._StoreTrueAction)
                default_value = False if is_true_action else True
                
                metadata = {}
                if include_help and action.help:
                    metadata['help'] = action.help
                if include_required:
                    metadata['required'] = action.required
                    
                section[arg_name] = {
                    'value': default_value,
                    'metadata': metadata
                }
    
    return config_data


def _get_arg_name(action: argparse.Action) -> Optional[str]:
    """Get the canonical name for an argument."""
    if action.dest == 'help' or action.dest == argparse.SUPPRESS:
        return None
    
    # Use the longest option without '--' prefix
    if action.option_strings:
        options = sorted(action.option_strings, key=len, reverse=True)
        for opt in options:
            if opt.startswith('--'):
                return opt[2:]  # Remove leading --
        return options[0].lstrip('-')
    
    return action.dest


def _write_yaml_config(
    config_data: Dict[str, Any],
    output_file: str,
    include_help: bool,
    include_choices: bool,
    include_required: bool
) -> None:
    """Write configuration data to a YAML file with comments."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for YAML output. Install with 'pip install pyyaml'.")
    
    # Process the config data to add comments
    yaml_lines = []
    
    def process_section(section, level=0):
        section_lines = []
        indent = '  ' * level
        
        for key, item in section.items():
            if isinstance(item, dict) and 'value' in item and 'metadata' in item:
                # Add comments for metadata
                metadata = item['metadata']
                if include_help and 'help' in metadata and metadata['help']:
                    section_lines.append(f"{indent}# {metadata['help']}")
                if include_choices and 'choices' in metadata and metadata['choices']:
                    section_lines.append(f"{indent}# Choices: {', '.join(str(c) for c in metadata['choices'])}")
                if include_required and 'required' in metadata:
                    section_lines.append(f"{indent}# Required: {metadata['required']}")
                
                # Add the actual value
                value = item['value']
                if value is None:
                    section_lines.append(f"{indent}{key}: null")
                elif isinstance(value, bool):
                    section_lines.append(f"{indent}{key}: {str(value).lower()}")
                elif isinstance(value, (int, float)):
                    section_lines.append(f"{indent}{key}: {value}")
                elif isinstance(value, list):
                    section_lines.append(f"{indent}{key}: {value}")
                else:
                    section_lines.append(f"{indent}{key}: '{value}'")
            elif isinstance(item, dict) and 'value' not in item:
                # This is a nested section
                section_lines.append(f"{indent}{key}:")
                nested_lines = process_section(item, level + 1)
                section_lines.extend(nested_lines)
        
        return section_lines
    
    # Process the root level
    for key, item in config_data.items():
        yaml_lines.append(f"# {key} Section")
        if isinstance(item, dict) and 'value' not in item:
            yaml_lines.append(f"{key}:")
            nested_lines = process_section(item, 1)
            yaml_lines.extend(nested_lines)
        else:
            nested_lines = process_section({key: item}, 0)
            yaml_lines.extend(nested_lines)
        yaml_lines.append("")  # Add blank line between sections
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write("\n".join(yaml_lines))


def _write_json_config(config_data: Dict[str, Any], output_file: str) -> None:
    """Write configuration data to a JSON file."""
    # Simplify the structure for JSON (remove metadata)
    simplified_data = {}
    
    def simplify_section(section):
        result = {}
        for key, item in section.items():
            if isinstance(item, dict) and 'value' in item:
                result[key] = item['value']
            elif isinstance(item, dict) and 'value' not in item:
                result[key] = simplify_section(item)
            else:
                result[key] = item
        return result
    
    simplified_data = simplify_section(config_data)
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(simplified_data, f, indent=2)


def _write_ini_config(
    config_data: Dict[str, Any],
    output_file: str,
    include_help: bool,
    include_choices: bool,
    include_required: bool
) -> None:
    """Write configuration data to an INI file with comments."""
    config = configparser.ConfigParser()
    
    # Process each section
    for section_name, section in config_data.items():
        if not isinstance(section, dict) or ('value' in section and 'metadata' in section):
            # This is not a proper section, create a DEFAULT section
            if 'DEFAULT' not in config:
                config['DEFAULT'] = {}
            if isinstance(section, dict) and 'value' in section:
                config['DEFAULT'][section_name] = str(section['value'] if section['value'] is not None else '')
            else:
                config['DEFAULT'][section_name] = str(section if section is not None else '')
        else:
            # This is a proper section
            if section_name not in config:
                config[section_name] = {}
            
            for key, item in section.items():
                if isinstance(item, dict) and 'value' in item:
                    value = item['value']
                    config[section_name][key] = str(value if value is not None else '')
                else:
                    config[section_name][key] = str(item if item is not None else '')
    
    # Write to file
    with open(output_file, 'w') as f:
        config.write(f)
    
    # Add comments (configparser doesn't support comments directly)
    if include_help or include_choices or include_required:
        with open(output_file, 'r') as f:
            ini_lines = f.readlines()
        
        # Process comments
        new_lines = []
        current_section = 'DEFAULT'
        
        for line in ini_lines:
            if line.strip().startswith('[') and line.strip().endswith(']'):
                current_section = line.strip()[1:-1]
                new_lines.append(line)
            elif '=' in line:
                key = line.split('=')[0].strip()
                
                # Add comments
                comments = []
                section_data = config_data.get(current_section, {})
                item_data = section_data.get(key, {})
                
                if isinstance(item_data, dict) and 'metadata' in item_data:
                    metadata = item_data['metadata']
                    if include_help and 'help' in metadata and metadata['help']:
                        comments.append(f"; Help: {metadata['help']}")
                    if include_choices and 'choices' in metadata and metadata['choices']:
                        comments.append(f"; Choices: {', '.join(str(c) for c in metadata['choices'])}")
                    if include_required and 'required' in metadata:
                        comments.append(f"; Required: {metadata['required']}")
                
                if comments:
                    new_lines.extend([c + '\n' for c in comments])
                
                new_lines.append(line)
            else:
                new_lines.append(line)
        
        # Write updated content
        with open(output_file, 'w') as f:
            f.writelines(new_lines)


def load_config_to_parser(
    parser: argparse.ArgumentParser,
    config_file: str,
    format: str = None
) -> argparse.Namespace:
    """
    Load a configuration file and update the parser's default values.
    
    Args:
        parser: The ArgumentParser object to update
        config_file: Path to the configuration file
        format: Format of the config file (auto-detected if None)
        
    Returns:
        Namespace with loaded configuration values
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    # Auto-detect format if not specified
    if format is None:
        ext = os.path.splitext(config_file)[1].lower()
        if ext in ['.yml', '.yaml']:
            format = 'yaml'
        elif ext == '.json':
            format = 'json'
        elif ext in ['.ini', '.cfg', '.conf']:
            format = 'ini'
        else:
            raise ValueError(f"Cannot auto-detect format for {config_file}. Please specify format.")
    
    # Load config based on format
    if format.lower() == 'yaml':
        try:
            import yaml
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML is required for YAML files. Install with 'pip install pyyaml'.")
    elif format.lower() == 'json':
        with open(config_file, 'r') as f:
            config_data = json.load(f)
    elif format.lower() == 'ini':
        config = configparser.ConfigParser()
        config.read(config_file)
        config_data = {s: dict(config.items(s)) for s in config.sections()}
        # Add DEFAULT section
        if 'DEFAULT' in config:
            config_data['DEFAULT'] = dict(config.items('DEFAULT'))
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Flatten nested dictionaries
    flat_config = {}
    
    def flatten_dict(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, dict) and 'value' in v:
                flat_config[k] = v['value']
            elif isinstance(v, dict) and 'value' not in v and not any(kk in v for kk in ['metadata']):
                flatten_dict(v, f"{k}_" if prefix == '' else f"{prefix}{k}_")
            else:
                flat_config[f"{prefix}{k}"] = v
    
    flatten_dict(config_data)
    
    # Update parser defaults
    parser.set_defaults(**{k: v for k, v in flat_config.items() if v is not None})
    
    # Parse empty args to get the namespace with updated defaults
    return parser.parse_args([])


# Example usage
if __name__ == "__main__":
    # Create a sample parser


    args, parser = Composer_ArgsPaeser()
    # Generate config files in different formats
    generate_config_from_parser(parser, "config.yaml", format="yaml")
    generate_config_from_parser(parser, "config.json", format="json")
    generate_config_from_parser(parser, "config.ini", format="ini")
    
    # Load configuration and parse args
    args = load_config_to_parser(parser, "config.yaml", format="yaml")
    print(f"Loaded configuration: {args}")
