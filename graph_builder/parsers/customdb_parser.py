from pathlib import Path
import pandas as pd
import logging
import re
from typing import List
from graph_builder.parsers.base_parser import (
    BaseParser,
    BaseConfig,
    Relation,
    check_relation_types,
    check_entity_ids,
    is_biomedgps_format,
)

logger = logging.getLogger("graph_builder.parsers.customdb_parser")


class CustomdbParser(BaseParser):
    """A parser for CustomDB database, such as any database that is in the BioMedGPS format."""

    def __init__(
        self,
        reference_entity_file: Path,
        output_directory: Path,
        skip: bool = True,
        num_workers: int = 20,
        relation_type_dict_df: pd.DataFrame | None = None,
        relation_file: Path | None = None,
        **kwargs,
    ):
        if relation_file is None:
            raise ValueError("The relation file is not provided, it's required for the CustomDB parser.")

        config = BaseConfig(
            downloads=[],
            database="customdb",
        )

        super().__init__(
            reference_entity_file,
            None,
            output_directory,
            config,
            False,
            skip,
            num_workers,
            relation_type_dict_df,
            relation_file=relation_file,
        )

    def read_customdb(self, customdb_filepath: Path) -> pd.DataFrame:
        # Specify the column names
        df = pd.read_csv(
            customdb_filepath,
            sep="\t",
            dtype=str,
        )

        if not is_biomedgps_format(df):
            raise ValueError(
                "The file is not in the BioMedGPS format. Please make sure the file has the following columns: source_id, source_type, target_id, target_type, relation_type, resource, key_sentence."
            )

        errors = check_entity_ids(df["source_id"].to_list())
        if len(errors) > 0:
            # raise ValueError(f"The source_id column contains invalid ids. {errors}")
            # Filter out all the rows with invalid source_id
            logger.warning(f"The source_id column contains invalid ids. {errors}")
            df = df[~df["source_id"].isin(errors)]

        errors = check_entity_ids(df["target_id"].to_list())
        if len(errors) > 0:
            logger.warning(f"The target_id column contains invalid ids. {errors}")
            df = df[~df["target_id"].isin(errors)]

        errors = check_relation_types("relation_type", df["relation_type"].to_list())
        if len(errors) > 0:
            logger.warning(f"The relation_type column contains invalid relation types. {errors}")
            df = df[~df["relation_type"].isin(errors)]

        return df

    def get_value(self, row: pd.Series, column: str) -> str:
        item = row[column].values.tolist()
        if len(item) == 0:
            return ""
        else:
            return item[0]

    def extract_relations(self) -> List[Relation]:
        if self.relation_file is None:
            raise ValueError("The relation file is not provided.")

        relations = self.read_customdb(self.relation_file)
        logger.info("Get %d relations" % len(relations))

        # These relations might be lacking key_sentence or pmids columns, so we need to add them if they are missing.
        if "key_sentence" not in relations.columns:
            relations["key_sentence"] = ""

        if "pmids" not in relations.columns:
            relations["pmids"] = ""

        return [Relation.from_args(**row) for row in relations.to_dict(orient="records")]  # type: ignore


if __name__ == "__main__":
    import logging
    import coloredlogs
    import verboselogs

    logging.basicConfig(level=logging.DEBUG)
    verboselogs.install()
    # Use the logger name instead of the module name
    coloredlogs.install(
        fmt="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
    )

    parser = CustomdbParser(
        Path("/Volumes/ProjectData/Current/Datasets/biomedgps/graph_data/entities.tsv"),
        Path("/Users/jy006/Downloads/Development/biomedgps"),
        relation_file=Path(
            "/Users/jy006/Downloads/Development/biomedgps/relations.tsv"
        ),
    )

    parser.parse()
