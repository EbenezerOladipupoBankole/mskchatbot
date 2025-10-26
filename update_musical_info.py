import json
import os

def update_hymn_musical_info():
    # Read the existing hymn book
    with open('hymn_book.json', 'r', encoding='utf-8') as f:
        hymns = json.load(f)

    # Musical information for each hymn with enhanced details
    hymn_musical_info = {
        "1": {  # The Morning Breaks
            "key_signature": "Ab Major",
            "time_signature": "4/4",
            "tempo": "Majestically",
            "meter": "8.8.8.8.8.",
            "musical_setting": {
                "harmonization": "SATB",
                "ranges": {
                    "soprano": "Eb4-F5",
                    "alto": "Ab3-C5",
                    "tenor": "Ab3-Db4",
                    "bass": "Db3-Ab3"
                }
            },
            "dynamics": {
                "opening": "mf",
                "markings": ["cresc.", "f", "dim.", "mp"],
                "expression": "With dignity and strength"
            },
            "musical_elements": {
                "form": "Strophic",
                "texture": "Homophonic",
                "harmony": {
                    "primary_chords": ["Ab", "Db", "Eb"],
                    "cadence": "Perfect Authentic"
                },
                "rhythmic_features": ["Dotted quarter notes", "Syncopation"],
                "melodic_character": "Ascending melodic line, stepwise motion"
            }
        },
        "2": {  # The Spirit of God
            "key_signature": "D Major",
            "time_signature": "2/2",
            "tempo": "With vigor",
            "meter": "13.13.13.13.13.13.",
            "musical_setting": {
                "harmonization": "SATB",
                "ranges": {
                    "soprano": "D4-G5",
                    "alto": "A3-D5",
                    "tenor": "F#3-E4",
                    "bass": "D3-A3"
                }
            },
            "dynamics": {
                "opening": "f",
                "markings": ["ff", "mf", "cresc.", "poco rit."],
                "expression": "With enthusiasm and joy"
            },
            "musical_elements": {
                "form": "Verse-Chorus",
                "texture": "Homophonic with some polyphonic elements",
                "harmony": {
                    "primary_chords": ["D", "G", "A", "Bm"],
                    "cadence": "Perfect Authentic"
                },
                "rhythmic_features": ["Strong downbeats", "March-like rhythm"],
                "melodic_character": "Bold, ascending phrases with repeated motifs"
            }
        },
        "3": {  # Now Let Us Rejoice
            "key_signature": "G Major",
            "time_signature": "6/8",
            "tempo": "Joyfully",
            "meter": "11.11.11.11",
            "musical_setting": {
                "harmonization": "SATB",
                "ranges": {
                    "soprano": "D4-E5",
                    "alto": "B3-B4",
                    "tenor": "D3-G4",
                    "bass": "G2-D4"
                }
            },
            "dynamics": {
                "opening": "mf",
                "markings": ["f", "mp", "cresc.", "dim."],
                "expression": "With brightness and joy"
            },
            "musical_elements": {
                "form": "Strophic",
                "texture": "Homophonic",
                "harmony": {
                    "primary_chords": ["G", "C", "D", "Em"],
                    "cadence": "Perfect Authentic"
                },
                "rhythmic_features": ["Compound meter", "Lilting rhythm"],
                "melodic_character": "Flowing, pastoral quality"
            }
        },
        "4": {  # Truth Eternal
            "key_signature": "Bb Major",
            "time_signature": "4/4",
            "tempo": "With dignity",
            "meter": "8.7.8.7.D",
            "musical_setting": {
                "harmonization": "SATB",
                "ranges": {
                    "soprano": "Bb3-F5",
                    "alto": "F3-D5",
                    "tenor": "Bb2-G4",
                    "bass": "Eb2-D4"
                }
            },
            "dynamics": {
                "opening": "mp",
                "markings": ["mf", "f", "dim.", "rit."],
                "expression": "Thoughtfully and reverently"
            },
            "musical_elements": {
                "form": "Binary",
                "texture": "Homophonic",
                "harmony": {
                    "primary_chords": ["Bb", "Eb", "F", "Gm"],
                    "cadence": "Perfect Authentic"
                },
                "rhythmic_features": ["Quarter note pulse", "Occasional dotted rhythms"],
                "melodic_character": "Gentle, arching phrases"
            }
        },
        "5": {  # High on the Mountain Top
            "key_signature": "Eb Major",
            "time_signature": "4/4",
            "tempo": "Boldly",
            "meter": "6.6.6.6.6.6",
            "musical_setting": {
                "harmonization": "SATB",
                "ranges": {
                    "soprano": "Eb4-Eb5",
                    "alto": "Bb3-C5",
                    "tenor": "G3-G4",
                    "bass": "Eb2-Eb4"
                }
            },
            "dynamics": {
                "opening": "f",
                "markings": ["ff", "mf", "cresc.", "maestoso"],
                "expression": "With majesty and strength"
            },
            "musical_elements": {
                "form": "Strophic with refrain",
                "texture": "Homophonic",
                "harmony": {
                    "primary_chords": ["Eb", "Ab", "Bb", "Cm"],
                    "cadence": "Perfect Authentic"
                },
                "rhythmic_features": ["Strong march-like rhythm", "Dotted quarters"],
                "melodic_character": "Bold, proclamatory style"
            }
        },
        # Add more hymns here...
    }

    # Update each hymn with musical information
    for hymn in hymns:
        hymn_num = hymn['number']
        if hymn_num in hymn_musical_info:
            hymn.update({
                "musical_info": hymn_musical_info[hymn_num]
            })
        else:
            # Default musical info template for hymns we don't have specific data for yet
            hymn.update({
                "musical_info": {
                    "key_signature": "To be added",
                    "time_signature": "To be added",
                    "tempo": "To be added",
                    "meter": "To be added",
                    "musical_setting": {
                        "harmonization": "SATB",
                        "ranges": {
                            "soprano": "To be added",
                            "alto": "To be added",
                            "tenor": "To be added",
                            "bass": "To be added"
                        }
                    }
                }
            })

    # Save the updated hymn book
    with open('hymn_book.json', 'w', encoding='utf-8') as f:
        json.dump(hymns, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    update_hymn_musical_info()
    print("✅ Hymn book updated with musical information!")